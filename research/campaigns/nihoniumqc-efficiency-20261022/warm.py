"""Mutate two known CPM parents; preserve every official search witness."""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import run as core

ROOT = core.ROOT
HERE = Path(__file__).resolve().parent
SEED = 2026102202


def main():
    out = HERE / "warm"
    out.mkdir(exist_ok=True)
    if (out / "results.json").exists():
        raise RuntimeError("Warm batch already recorded; do not repeat it.")
    fs = [core.forms(t) for t in [2, 3, 4]]
    blocks = []
    for f in fs:
        z = np.zeros_like(f)
        blocks.extend([np.hstack([f, z]), np.hstack([z, f])])
    mat = np.vstack(blocks)
    boundary = 2 * (len(fs[0]) + len(fs[1]))
    parents = {}
    for p, name in [(109, "872-222-20"), (113, "904-230-22")]:
        doc = json.loads((ROOT / "codes" / f"{name}.json").read_text())
        es = []
        for side in ["X", "Z"]:
            e = np.zeros((3, 8), dtype=np.int64)
            for i in range(3):
                for q in doc["checks"][side][i * p]:
                    e[i, q // p] = (-q) % p
            assert [np.flatnonzero(r).tolist() for r in core.lift(e, p)] == doc["checks"][side]
            es.append(e)
        x = np.concatenate([e.ravel() for e in es])
        assert not np.any((core.equations() @ x) % p)
        parents[p] = (name, x, np.array(core.nullspace_mod(core.equations(), p)[0]))
    rng = np.random.default_rng(SEED)
    records = []
    for draw in range(12):
        start = time.monotonic()
        p = [109, 113][draw % 2]
        name, original, basis = parents[p]
        delta = (mat @ basis.T) % p
        active = np.flatnonzero(np.any(delta != 0, axis=0))
        x = original.copy()
        for _ in range(2):
            x = (x + int(rng.integers(1, p)) * basis[int(rng.choice(active))]) % p
        vals = (mat @ x) % p

        def score(v):
            return int(np.count_nonzero(v[:boundary] == 0)), int(np.count_nonzero(v[boundary:] == 0))

        objective = score(vals)
        for _ in range(2000):
            j = int(rng.choice(active))
            a = int(rng.integers(1, p))
            trial = (vals + a * delta[:, j]) % p
            s = score(trial)
            if s <= objective:
                x, vals, objective = (x + a * basis[j]) % p, trial, s
        digest = hashlib.sha256(x.tobytes() + str(p).encode()).hexdigest()
        row = {
            "draw": draw,
            "P": p,
            "parent": name,
            "objective": objective,
            "exponent_sha256": digest,
            "EX": x[:24].reshape(3, 8).tolist(),
            "EZ": x[24:].reshape(3, 8).tolist(),
            "status": "short_cycle_filter",
        }
        if not objective[0]:
            hx, hz = core.lift(x[:24].reshape(3, 8), p), core.lift(x[24:].reshape(3, 8), p)
            seed = SEED + 2000 + draw
            found = core.distance_rand_witness(hx, hz, 4000, seed, backend="fast", threads=8, pair_depth=8)
            assert found
            doc = core.make_submission(
                hx,
                hz,
                name=f"Warm CPM lift {p} draw {draw}",
                construction=f"CPM cube matching mutation of {name}; EX={row['EX']}; EZ={row['EZ']}",
                authors=["@nihoniumqc"],
                family="pair-partition-cpm",
                references=["arXiv:2607.14091", "arXiv:2609.30069"],
                trials=4,
                seed=seed,
                date="2026-10-22",
            )
            row["initial_witnesses"] = json.loads(json.dumps(doc["distance"]))
            doc["distance"][found.side].update(
                value=found.weight,
                witness=found.support,
                witness_provenance={
                    "found_by": ["@nihoniumqc"],
                    "date": "2026-10-22",
                    "tool": "research/kit/surrogate.distance_rand_witness (fast)",
                    "seeds": [seed],
                    "found_at_samples": 4000,
                    "survived_samples": 4000,
                },
            )
            doc["distance"]["d"] = min(doc["distance"]["X"]["value"], doc["distance"]["Z"]["value"])
            doc["provenance"].update(
                origin="submission",
                model="GPT-6.1",
                novelty="unknown",
                notes=f"Screen only. Parent {name}; neither exact distance nor novelty established.",
            )
            path = core.unique_path(
                str(
                    ROOT
                    / "research/candidates/nihoniumqc-20261022"
                    / f"{doc['n']}-{doc['k']}-{doc['distance']['d']}-warm-{draw}.json"
                ),
                doc,
            )
            errors = core.save_submission(doc, path)
            if errors:
                raise RuntimeError(f"Could not persist witness: {errors}")
            row.update(
                n=doc["n"],
                k=doc["k"],
                d_ub=doc["distance"]["d"],
                score_upper_bound=doc["k"] * doc["distance"]["d"] ** 2 / doc["n"],
                screen_witnesses=doc["distance"],
                trials_per_side=4000,
                seed=seed,
                path=str(Path(path).relative_to(ROOT)),
                status="screened_not_gated",
            )
        row["seconds"] = time.monotonic() - start
        records.append(row)
        (out / "results.json").write_text(json.dumps(records, indent=2) + "\n")
        print(
            json.dumps(
                {
                    k: v
                    for k, v in row.items()
                    if k in ["draw", "P", "n", "k", "d_ub", "score_upper_bound", "status", "seconds"]
                }
            ),
            flush=True,
        )
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "upstream_commit": "2ac3b244009edf688c3472a18d85bcd235e97e63",
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "seed": SEED,
                "draws": 12,
                "parents": [v[0] for v in parents.values()],
                "moves": 2000,
                "initial_mutations": 2,
                "trials_per_side": 4000,
                "pair_depth": 8,
                "threads": 8,
                "numpy_version": np.__version__,
                "status": "completed",
                "validated_find_count": 0,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
