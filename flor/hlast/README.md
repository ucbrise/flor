# Hindsight Logging Across Space and Time

## Log Propagators

### dmp-propagate.py
Based on Google's [diff-match-patch](https://github.com/google/diff-match-patch) libraries.

Here's an example command that should Just Work™
```
python dmp-propagate.py --in-dir tests/toy-simple --log-version 2 --out-dir out/dmp/toy-simple --log=DEBUG
```

### gt-propagate.py
Based on [Fine-grained and accurate source code differencing](https://doi.org/10.1145/2642937.2642982) GumTree algorithm.

Takes a line number and source file (the statement to propagate) and a target (a different version without).
```
python gt-propagate.py <lineno> <source> <target> [--out <result>]
```

To run over a set of tests, use the bash script and the codebase/file/version you want to test.
```
bash gt-propagate.sh [codebase [filename [version [gt-propagate-args]]]]
```

Finally, run `gt-outputs.sh` with no arguments to replicate the saved outputs.

## Testing
Pass in two different directories with the same filenames, and the test harness will compare their log outputs:
```
VERBOSE=1 ./test-harness.sh tests/toy-simple/v1-gt results/dmp/toy-simple/v1
```

(you can turn off VERBOSE if you want less output)

## Data Format
Data is organized as follows (see `tests/toy-simple/`):
```
tests/{{codebase}}/v{{version}}/{{filename}}.py
```
e.g.
* codebase: `toy-simple` (`toy`: synthetic, `real`: ecological)
* version: 2 (should be numeric, ascending, and without gaps)
* filename: `linear-regression-example.py` (this can be whatever)

If you want to propagate a log backwards from, say, `v4` we expected that there should also be a corresponding directory
that contains the logging version named `v4-log`, while previous versions should have a `v*-gt` with the Ground Truth.