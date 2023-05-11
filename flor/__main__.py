import sys
import time
from tqdm import tqdm
import subprocess
import os
import sys

from . import database

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
            jobid, path, script, args = database.step_worker(db_conn, pid, tstamp)
            if jobid is None:
                time.sleep(2)
            else:
                assert jobid is not None
                assert path is not None
                assert script is not None
                assert args is not None

                s = f"python {script} --flor BATCH {args}"

                my_env = os.environ.copy()
                my_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  # type: ignore
                try:
                    print(s)
                    subprocess.run(s.split(), cwd=path, env=my_env)
                except Exception as e:
                    print("subprocess exception", e)
                database.finish_job(db_conn, jobid)

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
