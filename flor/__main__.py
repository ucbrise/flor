import sys
import time
from tqdm import tqdm

from . import database

db_conn = database.start_db()


arg = sys.argv[1]

if arg == "status":
    if len(sys.argv) == 2:
        database.server_status(db_conn)
    elif len(sys.argv) > 2:
        database.server_status(db_conn, "COMPLETED")
elif arg == "server":
    print("FLOR: Server starting...")

    def gen8r():
        i = 0
        while True:
            yield i
            i += 1

    try:
        database.server_active(db_conn)
        for each in tqdm(gen8r()):
            time.sleep(2)
    except KeyboardInterrupt:
        print("Cleaning up...")
    except Exception as e:
        if not isinstance(e, KeyboardInterrupt):
            print(e)
        else:
            print("Cleaning up...")
    finally:
        database.server_completed(db_conn)
        db_conn.close()

elif arg == "shell":
    pass
