import git
import os
import sys

class Versioner:

    def __init__(self, exe_full_path):
        """
        :param exe_full_path: The full path to the executable
        """
        assert os.path.splitext(os.path.basename(exe_full_path))[1] == '.py'
        self.current_dir = os.path.dirname(exe_full_path)
        self.versioning_dir = Versioner.get_ancestor_repo_path(self.current_dir)

        if not self.versioning_dir:
            print("The following dir is not a git repo: {}\n".format(self.current_dir),
                  "Please initialize a git repo with a `.gitignore` file before continuing")

            while True:
                quit_flag = input("Would you like to quit? [y/N] ").strip()
                if quit_flag == 'N':
                    break
                elif quit_flag.lower() == 'y':
                    print("Quitting...")
                    sys.exit(0)
                else:
                    print('Invalid entry: {}'.format(quit_flag))

        # self.targetbasis = tempfile.mkdtemp(prefix='git.florist')
        # print("Target directory at: {}".format(self.targetbasis))

    @staticmethod
    def __is_git_repo__(path):
        # https://stackoverflow.com/a/39956572/9420936
        try:
            _ = git.Repo(path).git_dir
            return True
        except:
            return False

    @staticmethod
    def get_ancestor_repo_path(path):
        if not path or path == os.path.sep:
            return None
        path = os.path.abspath(path)
        if Versioner.__is_git_repo__(path):
            return path
        head, tail = os.path.split(path)
        return Versioner.get_ancestor_repo_path(head)

    def save_commit_event(self, msg):
        if not self.versioning_dir:
            # TODO: ADD GIT_IGNORE FOR THE USER
            repo = git.Repo.init(self.current_dir)
            repo.git.add(A=True)
            repo.index.commit(msg)
        else:
            repo = git.Repo(self.versioning_dir)
            repo.git.add(A=True)
            repo.index.commit(msg)

    def reset_hard(self):
        if not self.versioning_dir:
            repo = git.Repo(self.current_dir)
            repo.git.reset('--hard')
        else:
            repo = git.Repo(self.versioning_dir)
            repo.git.reset('--hard')

    # def protect_assets(self):
    #     shutil.copytree(os.path.join(self.versioning_dir, '.git'),
    #                     os.path.join(self.targetbasis, '.git'))
    #
    #     shutil.copy2(os.path.join(self.versioning_dir, '.gitignore'),
    #                  os.path.join(self.targetbasis, '.gitignore'))
    #
    # def restore_assets(self):
    #     shutil.copytree(os.path.join(self.targetbasis, '.git'),
    #                     os.path.join(self.versioning_dir, '.git'))
    #
    #     shutil.copy2(os.path.join(self.versioning_dir, '.gitignore'),
    #                  os.path.join(self.targetbasis, '.gitignore'))

