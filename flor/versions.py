from git.repo import Repo
from git.exc import InvalidGitRepositoryError

import os

CURRDIR = os.getcwd()
SHADOW_BRANCH_PREFIX = "flor."


def get_repo_dir():
    try:
        repo = Repo(CURRDIR, search_parent_directories=True)
        return repo.working_dir
    except InvalidGitRepositoryError:
        print("Not a valid Git repository")
    except Exception as e:
        print(f"An error occurred while getting the repository directory: {e}")


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
            print("Changes committed successfully")
        else:
            print("No changes to commit")
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
            base_shadow_name = "flor.shadow"
            new_branch_name = base_shadow_name
            suffix = 1

            # Check if the branch name exists and increment the suffix until a unique name is found
            while any(b.name == new_branch_name for b in repo.branches):
                new_branch_name = f"{base_shadow_name}{suffix}"
                suffix += 1

            # Create a new branch with the unique name
            repo.git.checkout("-b", new_branch_name)
            print(f"Created and switched to new branch: {new_branch_name}")
    except InvalidGitRepositoryError:
        print("Not a valid Git repository")
    except Exception as e:
        print(f"An error occurred while processing the branch: {e}")


def get_latest_autocommit():
    try:
        repo = Repo(CURRDIR, search_parent_directories=True)
        for v in repo.iter_commits():
            if str(v.message).count("FLOR::") == 1:
                _, _, ts = v.message.strip().split("::")  # type: ignore
                yield (
                    str(ts),
                    v.hexsha,
                    v.authored_datetime.isoformat(timespec="seconds")[0 : len(ts)],
                )
    except InvalidGitRepositoryError:
        print("Not a valid Git repository")
    except Exception as e:
        print(f"An error occurred while processing the branch: {e}")


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
