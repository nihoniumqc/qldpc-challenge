"""Bounded CPM exponent search; all distance work uses the upstream kit."""

import argparse
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT / "research"), str(ROOT / "research/kit"), str(ROOT / "verify")]
from build_halved_pp import nullspace_mod
from campaign import Ledger, load_campaign, scaffold_campaign, write_campaign, write_summary
from coordination import unique_path
from submit import make_submission, save_submission
from surrogate import distance_rand_witness

MATCHINGS = [[(0, 4), (1, 7), (2, 6), (3, 5)], [(0, 7), (1, 2), (3, 4), (5, 6)], [(0, 2), (1, 5), (3, 7), (4, 6)]]
LIFTS = [97, 101, 107, 113]
SEED = 2026102201


def equations():
    rows = []
    for i in range(3):
        for j in range(3):
            for a, b in MATCHINGS[(i - j) % 3]:
                r = np.zeros(48, dtype=np.int64)
                r[8 * i + a], r[8 * i + b] = 1, -1
                r[24 + 8 * j + a], r[24 + 8 * j + b] = -1, 1
                rows.append(r)
    return np.array(rows)


def forms(t):
    rows = [r for r in itertools.product(range(3), repeat=t) if all(r[i] != r[(i + 1) % t] for i in range(t))]
    cols = [c for c in itertools.product(range(8), repeat=t) if all(c[i] != c[(i + 1) % t] for i in range(t))]
    out = set()
    for r in rows:
        for c in cols:
            f = [0] * 24
            for i in range(t):
                f[8 * r[i] + c[i]] += 1
                f[8 * r[(i + 1) % t] + c[i]] -= 1
            first = next((v for v in f if v), 0)
            if first:
                out.add(tuple(f if first > 0 else [-v for v in f]))
    return np.array(sorted(out), dtype=np.int64)


def lift(e, p):
    h = np.zeros((3 * p, 8 * p), dtype=np.int8)
    r = np.arange(p)
    for i in range(3):
        for j in range(8):
            h[i * p + r, j * p + (r - int(e[i, j])) % p] = 1
    return h


def calibrate():
    data = {
        "EX": [[13, 82, 43, 52, 73, 10, 49, 97], [48, 26, 5, 31, 30, 71, 42, 68], [25, 95, 9, 6, 93, 0, 23, 74]],
        "EZ": [
            [106, 2, 90, 58, 57, 16, 96, 17],
            [101, 108, 22, 105, 83, 36, 59, 41],
            [101, 97, 58, 39, 60, 33, 72, 76],
        ],
    }
    doc = json.loads((ROOT / "codes/872-222-20.json").read_text())
    for key, ex in [("X", data["EX"]), ("Z", data["EZ"])]:
        h = lift(np.array(ex), 109)
        assert [np.flatnonzero(row).tolist() for row in h] == doc["checks"][key]


def climb(p, basis, mat, boundary, rng, moves):
    delta = (mat @ basis.T) % p
    active = np.flatnonzero(np.any(delta != 0, axis=0))
    x = (rng.integers(p, size=len(basis)) @ basis) % p
    vals = (mat @ x) % p

    def score(v):
        return int(np.count_nonzero(v[:boundary] == 0)), int(np.count_nonzero(v[boundary:] == 0))

    best = score(vals)
    for _ in range(moves):
        j = int(rng.choice(active))
        a = int(rng.integers(1, p))
        trial = (vals + a * delta[:, j]) % p
        s = score(trial)
        if s <= best:
            x, vals, best = (x + a * basis[j]) % p, trial, s
    return x, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=16)
    ap.add_argument("--moves", type=int, default=4000)
    ap.add_argument("--trials", type=int, default=4000)
    args = ap.parse_args()
    calibrate()
    campaign_path = HERE / "campaign.json"
    if campaign_path.exists():
        raise RuntimeError("This dated batch already exists; resume its saved candidates instead of repeating it.")
    spec = scaffold_campaign(
        HERE.name,
        "CPM efficiency search, 22 October 2026",
        budget={"candidates_screened": args.draws, "walltime_hours": 1},
        families=["other"],
        constraints={"n": [700, 1000], "max_check_weight": 8},
        notes="Maximize witness-based provisional k*d^2/n; only the unchanged trusted gate can accept a candidate.",
    )
    write_campaign(spec, str(campaign_path))
    led = Ledger(
        load_campaign(str(campaign_path)),
        journal=str(HERE / "journal.jsonl"),
        manifest=str(HERE / "manifest.json"),
        params={**vars(args), "seed": SEED, "lifts": LIFTS, "pair_depth": 8, "threads": 8, "numpy_trials_per_side": 4},
    )
    fs = [forms(t) for t in [2, 3, 4]]
    assert [len(f) for f in fs] == [84, 336, 4830]
    blocks = []
    for f in fs:
        z = np.zeros_like(f)
        blocks.extend([np.hstack([f, z]), np.hstack([z, f])])
    mat = np.vstack(blocks)
    boundary = 2 * (len(fs[0]) + len(fs[1]))
    bases = {p: np.array(nullspace_mod(equations(), p)[0]) for p in LIFTS}
    rng = np.random.default_rng(SEED)
    staged = ROOT / "research/candidates/nihoniumqc-20261022"
    staged.mkdir(parents=True, exist_ok=True)
    results = []
    for draw in range(args.draws):
        p = LIFTS[draw % len(LIFTS)]
        t0 = time.monotonic()
        x, objective = climb(p, bases[p], mat, boundary, rng, args.moves)
        digest = hashlib.sha256(x.tobytes() + str(p).encode()).hexdigest()
        if objective[0]:
            results.append(
                {
                    "draw": draw,
                    "P": p,
                    "exponent_sha256": digest,
                    "objective": objective,
                    "status": "short_cycle_filter",
                    "EX": x[:24].reshape(3, 8).tolist(),
                    "EZ": x[24:].reshape(3, 8).tolist(),
                }
            )
            (HERE / "screenings.json").write_text(json.dumps(results, indent=2) + "\n")
            print("FILTER", draw, p, objective, flush=True)
            continue
        ex, ez = x[:24].reshape(3, 8), x[24:].reshape(3, 8)
        hx, hz = lift(ex, p), lift(ez, p)
        seed = SEED + 1000 + draw
        led.start_experiment(
            "other",
            seed=seed,
            mode="novel_generation",
            params={
                "construction_family": "pair-partition-cpm",
                "P": p,
                "draw": draw,
                "exponent_sha256": digest,
                "EX": ex.tolist(),
                "EZ": ez.tolist(),
            },
        )
        found = distance_rand_witness(hx, hz, args.trials, seed, backend="fast", threads=8, pair_depth=8)
        assert found, "No logical witness returned by official search"
        doc = make_submission(
            hx,
            hz,
            name=f"CPM lift {p}, draw {draw}",
            construction=f"Pair-partition CPM (3,8), P={p}; EX={ex.tolist()}; EZ={ez.tolist()}",
            authors=["@nihoniumqc"],
            family="pair-partition-cpm",
            references=["arXiv:2607.14091", "arXiv:2609.30069"],
            trials=4,
            seed=seed,
            date="2026-10-22",
        )
        initial_witnesses = json.loads(json.dumps(doc["distance"]))
        doc["distance"][found.side].update(
            value=found.weight,
            witness=found.support,
            witness_provenance={
                "found_by": ["@nihoniumqc"],
                "date": "2026-10-22",
                "tool": "research/kit/surrogate.distance_rand_witness (fast)",
                "seeds": [seed],
                "found_at_samples": args.trials,
                "survived_samples": args.trials,
            },
        )
        doc["distance"]["d"] = min(doc["distance"]["X"]["value"], doc["distance"]["Z"]["value"])
        doc["provenance"].update(
            origin="submission",
            model="GPT-6.1",
            novelty="unknown",
            notes="Screening candidate only; not a validated find or exact-distance claim.",
        )
        path = unique_path(str(staged / f"{8 * p}-{doc['k']}-{doc['distance']['d']}-{draw}.json"), doc)
        errors = save_submission(doc, path)
        if errors:
            raise RuntimeError(f"Could not persist witness: {errors}")
        score = doc["k"] * doc["distance"]["d"] ** 2 / doc["n"]
        row = {
            "draw": draw,
            "P": p,
            "n": doc["n"],
            "k": doc["k"],
            "d_ub": doc["distance"]["d"],
            "score_upper_bound": score,
            "exponent_sha256": digest,
            "objective": objective,
            "trials_per_side": args.trials,
            "seed": seed,
            "path": str(Path(path).relative_to(ROOT)),
            "EX": ex.tolist(),
            "EZ": ez.tolist(),
            "initial_witnesses": initial_witnesses,
            "status": "screened_not_gated",
            "seconds": time.monotonic() - t0,
        }
        results.append(row)
        led.record_screen(trials=args.trials, d=doc["distance"]["d"], backend="fast", n=doc["n"], k=doc["k"])
        led.record_verdict("not_run")
        led.spend(candidates_screened=1)
        led.end_experiment()
        write_summary(led.summary(), str(HERE / "summary.json"))
        (HERE / "screenings.json").write_text(json.dumps(results, indent=2) + "\n")
        print(
            json.dumps({k: row[k] for k in ["draw", "P", "n", "k", "d_ub", "score_upper_bound", "seconds"]}), flush=True
        )
    write_summary(led.summary(status="completed"), str(HERE / "summary.json"))


if __name__ == "__main__":
    main()
