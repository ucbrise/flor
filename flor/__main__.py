import sys
import time
from tqdm import tqdm
import subprocess
import os
import sys

from . import database
from flor.query.engine import apply_variables

from git.repo import Repo

db_conn = database.start_db()


arg = sys.argv[1]

if arg == "status":
    if len(sys.argv) == 2:
        database.server_status(db_conn)
    elif len(sys.argv) > 2:
        database.server_status(db_conn, "COMPLETED")
elif arg == "serve":
    gpu_id = None
    if len(sys.argv) > 2:
        gpu_id = int(sys.argv[2])
        assert gpu_id >= 0
    print(
        f"FLOR: Server starting{ f' on GPU {str(gpu_id)}' if gpu_id is not None else ''}..."
    )

    def gen8r():
        i = 0
        while True:
            yield i
            i += 1

    try:
        pid, tstamp = database.server_active(db_conn, gpu_id)
        for each in tqdm(gen8r()):
            res = database.step_worker(db_conn, pid, tstamp)
            if res is not None:
                if res[0] == "jobs":
                    jobid, path, script, args = res[1]

                    s = f"python {script} --flor BATCH {args}"

                    my_env = os.environ.copy()
                    my_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  # type: ignore
                    try:
                        print("\n", s)
                        subprocess.run(s.split(), cwd=path, env=my_env)
                    except Exception as e:
                        print("subprocess exception", e)
                    database.finish_job(db_conn, jobid)
                elif res[0] == "replay":
                    jobid, path, script, vid, apply_vars, mode = res[1]
                    path = path.replace("\x1b[m", "")
                    s = f"python {script} {mode}"

                    my_env = os.environ.copy()
                    my_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  # type: ignore

                    repo = Repo(path)
                    current_commit = repo.head.commit
                    active_branch = repo.active_branch

                    try:
                        print("\n", s)
                        # TODO: git checkout VID
                        # print(path)
                        os.chdir(path)
                        repo.git.checkout(vid)
                        apply_variables(apply_vars.split(", "), script)
                        subprocess.run(s.split(), check=True, cwd=path, env=my_env)
                    except Exception as e:
                        print("subprocess exception", e)
                    finally:
                        repo.git.stash()
                        # repo.git.reset("--hard", current_commit)
                        repo.heads[active_branch.name].checkout()
                    database.finish_replay(db_conn, jobid)
                else:
                    raise
            else:
                time.sleep(2)
    except KeyboardInterrupt:
        print("Cleaning up...")
    except Exception as e:
        print(e)
    finally:
        database.server_completed(db_conn)
        db_conn.close()

elif arg == "config":
    # Assume default arguments
    pass
