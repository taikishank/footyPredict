# CLAUDE.md

## Running ML jobs (ml/tune.py, ml/*_ablate.py, ml/train_*.py)

This machine has only 4 CPU cores. Each of these scripts fits XGBoost models
and will use all available cores by default. Launching more than one at a
time does not parallelize the work — it oversubscribes the 4 cores (load
average spikes to ~16+), and every job slows down by 10-50x from contention
rather than finishing faster.

**Always run these jobs one at a time, sequentially** (e.g. chained with `&&`
or in a single `bash -c '...; ...; ...'` in the background), never launched
concurrently with `&` in the same shell. If several sweeps need to run,
queue them in one sequential chain and let it run unattended rather than
firing them off in parallel "to save time."
