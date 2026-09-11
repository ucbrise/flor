# Working on a FlorDB branch

FlorDB creates **and switches to** `flor.branch` on your first run from a branch
whose name doesn't start with `flor.`. If that name is taken, it uses
`flor.branch1`, `flor.branch2`, and so on. After the run, your checkout stays on
the new branch, so subsequent edits and commits belong there too. Your original
branch, such as `main`, stays at its previous commit.

Already managing your own experiment branches? Create and switch to a name
with the `flor.` prefix before running your script:

```bash
git switch -c flor.experiment
```

FlorDB honors that branch and keeps using it for auto-commits.

## Explore several leads

Give each lead its own `flor.` branch, created from wherever that lead starts:

```bash
git switch -c flor.lead-a
python train.py
git switch main
git switch -c flor.lead-b
python train.py
```

Each branch's history holds its runs and the code that produced them.

`flor.dataframe` reads the local cache, which isn't tracked and stays put when
you switch branches. `python -m flordb unpack` adds the checked-out branch's
runs to it, so to compare leads, unpack each one:

```bash
git switch flor.lead-a && python -m flordb unpack
git switch flor.lead-b && python -m flordb unpack
```

Replay finds runs through the current branch's history. To replay a run from
another lead, switch to that lead's branch first.

## Check what's saved

At the end of a run, FlorDB makes a local auto-commit containing the run record
and your code changes. It uses `git add -A`, so the commit includes all pending
changes Git can stage, including unignored new files and deletions. See
[Storage](storage.md) for which FlorDB files are tracked.

Check your current branch, any remaining edits, and the latest commit:

```bash
git branch --show-current
git status
git log -1 --stat
```

Edits you make after the run still need saving. You can commit them normally
without running another experiment. Replace `path/to/script.py` with the files
you want to save:

```bash
git add path/to/script.py
git commit -m "Refine experiment"
```

## Publish your branch

Auto-commits are local; FlorDB doesn't push them. With a remote named `origin`
already configured, publish your current branch and set its upstream:

```bash
git push -u origin HEAD
```

After more runs or manual commits, use `git push` to publish the new commits.
This shares the branch's committed code and run records. Checkpoint files and
the local database cache aren't included.

A teammate with a clone can fetch the branch, switch to it, and rebuild the
metrics cache. Substitute the branch name you published:

```bash
git fetch origin
git switch flor.experiment
python -m flordb unpack
```

## Bring code back for review

You can open a pull request from your published `flor.*` branch using your
normal Git hosting workflow. That includes its run history as well as its code.

To bring only selected code changes into a review branch, first commit any
remaining edits on your FlorDB branch. Then create a review branch from `main`
and copy the files you want from the experiment branch:

```bash
git switch main
git switch -c improve-training
git restore --source flor.experiment -- path/to/script.py
git diff
git add path/to/script.py
git commit -m "Improve training"
git push -u origin HEAD
```

Replace the example branch and file names with yours. `git restore` copies each
selected file's entire version from the experiment branch, so review the diff
before committing, especially if `main` has changed since the experiment began.
Your experiment branch retains its run history and remains available whenever
you want to return with `git switch flor.experiment`.
