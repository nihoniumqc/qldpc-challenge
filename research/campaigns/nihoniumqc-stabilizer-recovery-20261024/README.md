# Stabilizer campaign checkpoint

Author: @nihoniumqc. Checkpoint recovered on 24 October 2026.

The archive contains the complete 23 October stabilizer campaign: construction scripts, exponent arrays, matrices, packaged logical witnesses, seeds, search budgets, negative outcomes and recovery records. Extract it at the root of upstream unitaryfoundation/qldpc-challenge commit 20300a96a1d8ee4d766cf250352aaae14963a610. The archive includes its own README and reproduction scripts.

Completed work: 20 raw halved-CPM attempts (19 constructed, one exhausted), 12 optimized constructions and eight mirror folds: 39 codes screened. Each received 4,000 accelerated trials per doubled side and eight NumPy Pauli trials. Eight folds received an additional 100,000 accelerated trials per doubled side, 100 NumPy trials and a time-capped orbit/random-frame pass. The 284-qubit fold additionally completed 1,000,000 accelerated trials per doubled side and 1,000 NumPy trials.

| n | k | Best witnessed bound | k/n | Provisional kd²/n | Deepest accelerated rung per doubled side |
|---:|---:|---:|---:|---:|---:|
|284|73|d ≤ 17|0.257042|74.285211|1,000,000|
|436|111|d ≤ 18|0.254587|82.486239|100,000|
|404|103|d ≤ 20|0.254950|101.980198|100,000|
|456|117|d ≤ 22|0.256579|124.184211|100,000|

These are upper-bound search scores, not proven achieved efficiencies. The 284-qubit result establishes the first million-trial stabilizer baseline, so a percentage gain against a previous comparable stabilizer campaign is not applicable. CSS scores are a separate comparison class. The 452-qubit mirror of 904-230-22 is related to an existing entry and must not be refiled. Other equivalence and literature novelty remain unresolved.

On 24 October, current upstream was pinned to 47768db9e825eadc5ad55aad08a4936d37ca0f8b; its rule changes were reviewed and the unchanged trusted integrity check passed for 49 files. A new million-trial confirmation of the 456-qubit fold and a current candidate gate for the 284-qubit fold were launched. The execution environment became unavailable before their completion could be recovered. Neither is counted as completed validation. Resume from saved artifacts and inspect any surviving processes before restarting.

No stabilizer submission is authorized by these results alone: full current trusted submission validation, independent confirmation, frontier/dedup checks, and clean committed-tree publishing checks remain outstanding. No exact stabilizer distance certificate exists here.
