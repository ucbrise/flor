from git.repo import Repo
from git.exc import InvalidGitRepositoryError

import os

CURRDIR = os.getcwd()
SHADOW_BRANCH_PREFIX = "flor."
AUTO_COMMIT_SUBJECT_PREFIX = "FLOR::Auto-commit::"


def get_repo_dir():
    try:
        repo = Repo(CURRDIR, search_parent_directories=True)
        return repo.working_dir
    except InvalidGitRepositoryError:
        print("Not a valid Git repository")
    except Exception as e:
        print(f"An error occurred while getting the repository directory: {e}")


def ensure_gitignored(entry: str):
    repo_dir = get_repo_dir()
    if repo_dir is None:
        return
    gitignore_path = os.path.join(str(repo_dir), ".gitignore")
    entry = entry.strip()
    if os.path.exists(gitignore_path):
        with open(gitignore_path, "r") as f:
            lines = [line.strip() for line in f.readlines()]
        if entry in lines:
            return
        needs_newline = bool(lines) and lines[-1] != ""
        with open(gitignore_path, "a") as f:
            if needs_newline:
                f.write("\n")
            f.write(entry + "\n")
    else:
        with open(gitignore_path, "w") as f:
            f.write(entry + "\n")


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
            print("\nRun committed successfully.")
        else:
            print("\nNo changes to commit.")
    except InvalidGitRepositoryError:
        print("Not a valid Git repository")
    except Exception as e:
        print(f"An error occurred while committing: {e}")


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
                print(f"Created and switched to new branch: {new_branch_name}")
            except Exception as e:
                # Likely branch already exists due to race condition
                # repo.git.checkout(new_branch_name)
                branch = repo.active_branch.name
                print(
                    f"Branch '{new_branch_name}' already exists. Switched to branch: {branch}"
                )
    except InvalidGitRepositoryError:
        print("Not a valid Git repository")
    except Exception as e:
        print(f"An error occurred while processing the branch: {e}")


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
        print("Not a valid Git repository")
    except Exception as e:
        print(f"An error occurred while processing the branch: {e}")


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
    print("Checking out ", commit_hash)
    repo.git.checkout(commit_hash)


def get_head():
    repo = Repo(CURRDIR, search_parent_directories=True)
    current_head = repo.head.commit
    return current_head


def reset_hard():
    repo = Repo(CURRDIR, search_parent_directories=True)
    repo.git.reset("--hard")
