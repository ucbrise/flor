from git.repo import Repo
from git.exc import InvalidGitRepositoryError

import os

# capture imports nothing from flordb, so this is safe despite constants.py
# depending on this module.
from .capture import flor_print

CURRDIR = os.getcwd()
SHADOW_BRANCH_PREFIX = "flor."
AUTO_COMMIT_SUBJECT_PREFIX = "FLOR::Auto-commit::"


def get_repo_dir():
    try:
        repo = Repo(CURRDIR, search_parent_directories=True)
        return repo.working_dir
    except InvalidGitRepositoryError:
        flor_print("Not a valid Git repository")
    except Exception as e:
        flor_print(f"An error occurred while getting the repository directory: {e}")


# Everything under .flor/ is reconstructible except runs/. The sqlite cache is
# rebuilt by `flor unpack` and checkpoints are recomputed by replay, but a run's
# JSONL is the one irreplaceable observation, and it packs to ~69KB per run in
# git -- cheap enough to commit beside the code and args that produced it, so a
# clone or a `git fetch` carries the whole experiment history.
FLOR_IGNORE_ENTRIES = (".flor/*", "!.flor/runs/")

# Pre-v4 flor wrote a bare `.flor/`. Git does not descend into an excluded
# directory, so a `!.flor/runs/` added underneath it would never re-include
# anything -- the legacy line has to be removed, not just appended to.
LEGACY_FLOR_IGNORE_ENTRIES = (".flor/", ".flor")


def ensure_gitignored(entries=FLOR_IGNORE_ENTRIES, legacy=LEGACY_FLOR_IGNORE_ENTRIES):
    repo_dir = get_repo_dir()
    if repo_dir is None:
        return
    gitignore_path = os.path.join(str(repo_dir), ".gitignore")
    lines = []
    if os.path.exists(gitignore_path):
        with open(gitignore_path, "r") as f:
            lines = f.read().splitlines()

    kept = [line for line in lines if line.strip() not in legacy]
    present = {line.strip() for line in kept}
    missing = [e for e in entries if e not in present]
    if kept == lines and not missing:
        return

    # Appended at the end: gitignore is last-match-wins, so this keeps the
    # re-include from being undone by a broader pattern further down the file.
    out = kept + missing
    with open(gitignore_path, "w") as f:
        f.write("\n".join(out) + "\n")


def git_commit(message="FLOR::Auto-commit"):
    try:
        # Get the current working directory and initialize a Repo object
        repo = Repo(CURRDIR, search_parent_directories=True)

        # Check if there are any uncommitted changes
        if repo.is_dirty(untracked_files=True):
            # Add all untracked files and changes to tracked files
            repo.git.add(A=True)

            # Commit the changes
            repo.git.commit(m=message)
            flor_print("\nRun committed successfully.")
        else:
            flor_print("\nNo changes to commit.")
    except InvalidGitRepositoryError:
        flor_print("Not a valid Git repository")
    except Exception as e:
        flor_print(f"An error occurred while committing: {e}")


def current_branch():
    try:
        repo = Repo(CURRDIR, search_parent_directories=True)
        return repo.active_branch
    except InvalidGitRepositoryError:
        return None
    except TypeError:
        return None


def to_shadow():
    try:
        repo = Repo(CURRDIR, search_parent_directories=True)
        branch = repo.active_branch.name
        if branch.startswith(SHADOW_BRANCH_PREFIX):
            # Branch already has the 'flor.' prefix, continuing...
            return
        else:
            base_shadow_name = SHADOW_BRANCH_PREFIX + "branch"
            new_branch_name = base_shadow_name
            suffix = 1

            # Check if the branch name exists and increment the suffix until a unique name is found
            while any(b.name == new_branch_name for b in repo.branches):
                new_branch_name = f"{base_shadow_name}{suffix}"
                suffix += 1

            try:
                # Try to create a new branch with the unique name
                repo.git.checkout("-b", new_branch_name)
                flor_print(f"Created and switched to new branch: {new_branch_name}")
            except Exception as e:
                # Likely branch already exists due to race condition
                # repo.git.checkout(new_branch_name)
                branch = repo.active_branch.name
                flor_print(
                    f"Branch '{new_branch_name}' already exists. Switched to branch: {branch}"
                )
    except InvalidGitRepositoryError:
        flor_print("Not a valid Git repository")
    except Exception as e:
        flor_print(f"An error occurred while processing the branch: {e}")


def get_latest_autocommit():
    try:
        repo = Repo(CURRDIR, search_parent_directories=True)
        for v in repo.iter_commits():
            message = str(v.message)
            if AUTO_COMMIT_SUBJECT_PREFIX not in message:
                continue
            subject = message.strip().splitlines()[0]
            if not subject.startswith(AUTO_COMMIT_SUBJECT_PREFIX):
                continue
            ts = subject[len(AUTO_COMMIT_SUBJECT_PREFIX):]
            yield (
                str(ts),
                v.hexsha,
                v.authored_datetime.isoformat(timespec="seconds")[0 : len(ts)],
            )
    except InvalidGitRepositoryError:
        flor_print("Not a valid Git repository")
    except Exception as e:
        flor_print(f"An error occurred while processing the branch: {e}")


def read_args(commit_message: str) -> dict:
    """Parse k=v lines out of an auto-commit message body."""
    lines = commit_message.strip().splitlines()
    args = {}
    for line in lines[1:]:
        line = line.strip()
        if not line or "=" not in line:
            continue
        key, _, value = line.partition("=")
        args[key.strip()] = value.strip()
    return args


def checkout(commit_hash):
    repo = Repo(CURRDIR, search_parent_directories=True)
    # Checkout to the desired commit
    flor_print("Checking out ", commit_hash)
    repo.git.checkout(commit_hash)


def get_head():
    repo = Repo(CURRDIR, search_parent_directories=True)
    current_head = repo.head.commit
    return current_head


def reset_hard():
    repo = Repo(CURRDIR, search_parent_directories=True)
    repo.git.reset("--hard")
