from git.repo import Repo
from git.exc import InvalidGitRepositoryError
import os

from flor import flags
from flor.state import State

from flor.constants import *
from flor.logger import exp_json, log_records
from flor.shelf import home_shelf
from flor.query import database
from flor.skipblock.writeblock import WriteBlock
from pathlib import Path

from time import time
from datetime import datetime
import atexit
import pandas as pd

PATH = Path(".flor")


def get_projid():
    if State.common_dir is None:
        r = Repo()
        State.common_dir = Path(r.common_dir)
    return (
        os.path.basename(os.path.dirname(str(State.common_dir)))
        + "_"
        + str(State.active_branch)
    )


def in_shadow_branch():
    """
    Initialize
    """
    try:
        r = Repo()
        State.repo = r
        try:
            State.active_branch = str(r.active_branch)
        except TypeError:
            bs = State.repo.git.branch(
                "--contains", str(State.repo.head.commit.hexsha)
            ).split("\n")
            branches = [e.strip() for e in bs if "flor.shadow" in e]
            if len(branches) == 1:
                State.active_branch = branches.pop()
            else:
                raise NotImplementedError()
        cond = check_branch_cond()
        if cond:
            PATH.mkdir(exist_ok=True)
            get_projid()
        return cond
    except InvalidGitRepositoryError:
        return False
    except Exception as e:
        print(e)
        return False


def check_branch_cond():
    if State.repo is not None:
        return "RECORD::" in str(
            State.repo.head.commit.message
        ) or "flor.shadow" in State.repo.git.branch(
            "--contains", str(State.repo.head.commit.hexsha)
        )
    return False


def eval_seconds():
    try:
        State.seconds["EVAL"] = time() - State.seconds["EVAL"]  # type: ignore
    except Exception:
        State.seconds["EVAL"] = -1


@atexit.register
def flush():
    if flags.NAME and not flags.REPLAY:
        eval_seconds()
        path = home_shelf.close()
        cond = in_shadow_branch()
        projid = get_projid()

        assert cond
        log_records.flush(projid, str(State.timestamp))
        exp_json.put("PROJID", projid)
        exp_json.put("EPOCHS", WriteBlock.journal.get_iterations_count())
        exp_json.flush()
        repo = State.repo
        assert repo is not None
        repo.git.add("-A")
        repo.index.commit(f"RECORD::{flags.NAME}")
    elif flags.NAME and flags.REPLAY:
        eval_seconds()
        path = home_shelf.close()
        cond = in_shadow_branch()
        projid = get_projid()
        assert State.repo

        if State.db_conn is None:
            database.start_db(projid)
        assert State.db_conn

        # TODO: Write replay metadata (e.g. seconds)
        new_tstamp = str(datetime.now().isoformat())
        hexsha = None
        for v in State.repo.iter_commits():
            if str(v.message).count("RECORD::") == 1:
                hexsha = v.hexsha
                break
        assert hexsha is not None

        pd.DataFrame(
            [
                {
                    c: v
                    for c, v in zip(
                        list(DATA_PREP) + ["prep_secs", "eval_secs"],
                        [
                            projid,
                            flags.NAME,
                            new_tstamp,
                            hexsha,
                            float(State.seconds["PREP"]),  # type: ignore
                            float(State.seconds["EVAL"]),  # type: ignore
                        ],
                    )
                },
            ]
        ).to_sql("data_prep", con=State.db_conn, if_exists="append", index=False)

        if flags.PID.pid == 1 and flags.PID.ngpus == 1:
            data = []
            for i, epoch_secs in enumerate(State.seconds["EPOCHS"]):  # type: ignore
                epoch = i + 1
                data.append(
                    {
                        c: v
                        for c, v in zip(
                            list(OUTR_LOOP)
                            + [
                                "seconds",
                            ],
                            [
                                projid,
                                flags.NAME,
                                new_tstamp,
                                hexsha,
                                int(epoch),
                                float(epoch_secs),
                            ],
                        )
                    }
                )
            pd.DataFrame(data).to_sql(
                "outr_loop", con=State.db_conn, if_exists="append", index=False
            )

        assert cond
        for k in [k for k in exp_json.record_d if not k.isupper()]:
            log_records.put_dp(k, exp_json.record_d[k])
        log_records.flush(projid, str(State.timestamp))

    if State.db_conn:
        State.db_conn.commit()
        State.db_conn.close()
