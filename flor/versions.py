from .constants import *

import git
from git.repo import Repo


def git_commit(message="Auto-commit"):
    try:
        # Get the current working directory and initialize a Repo object
        repo = Repo(CURRDIR)

        # Check if there are any uncommitted changes
        if repo.is_dirty():
            # Add all untracked files and changes to tracked files
            repo.git.add(A=True)

            # Commit the changes
            repo.git.commit(m=message)
            print("Changes committed successfully")
        else:
            print("No changes to commit")
    except git.InvalidGitRepositoryError:
        print("Not a valid Git repository")
    except Exception as e:
        print(f"An error occurred while committing: {e}")


# Usage example
if __name__ == "__main__":
    # Your library code here

    # Commit the changes with a custom message
    git_commit("Commit message for this run")
