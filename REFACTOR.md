# Version 4

## Least Redundancy Overhaul

If a script performs io, like `print` or `logging`, FlorDB will automatically capture the output of subsequent runs, and also be able to parse and integrate historical logs of various formats and layouts. This means you can start using FlorDB with zero code changes, and it will automatically index and structure your logs for easy retrieval and analysis (with minimal duplication).

## LLM Ready Data Layout

Right now flor writes all the logs to a single JSON file, once per run, and each run that file is overwritten, simultaneously, flor writes to a sqlite database in the user's home directory `~/.flor/projid.db`. This was fine when we were interfacing with flor using just `flor.log` and `flor.dataframe`, but it can be a limitationlofor the LLM Access Path. Instead, we want a new data layout:

1. The files will live in a working tree (e.g. `.flor/runs/*.jsonl`) — easiest to grep. Should be auto-gitignored to avoid pollution. Data sync will be described separately.
2. The `flor.arg` should be stored in the git commit, e.g. seeds, so some semblance of reproducibility is possible if the log files are destroyed on merge.
3. Do away with the `~/.flor` state, keep project level scope.
4. the db is a query cache, and it should be stored in the project `.flor` directory.

## Data sync-ing

We can't really have the logs living in git, but we want some measure or reproducibility. That's a balancing act.