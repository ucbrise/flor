import sqlite3

import pytest

from flordb import database, orm


def log(name, value, ctx=None, tstamp="2026-05-22T10:00:00.000000", type=1):
    return orm.Log("proj", tstamp, "train.py", ctx, name, value, type)


def seg(name, iteration, value=None):
    return orm.Segment(name, iteration, value)


class TestUnpack:
    def test_tags_source(self, logs_db):
        conn, cursor = logs_db
        database.unpack([log("loss", 0.5)], cursor, source="forward")
        database.unpack([log("val_acc", 90)], cursor, source="replay")
        cursor.execute("SELECT value_name, source FROM logs ORDER BY value_name")
        assert cursor.fetchall() == [("loss", "forward"), ("val_acc", "replay")]

    def test_rejects_unknown_source(self, logs_db):
        _, cursor = logs_db
        with pytest.raises(ValueError):
            database.unpack([log("loss", 0.5)], cursor, source="guess")

    def test_empty_buffer_is_a_noop(self, logs_db):
        conn, cursor = logs_db
        database.unpack([], cursor)
        cursor.execute("SELECT COUNT(*) FROM logs")
        assert cursor.fetchone()[0] == 0

    def test_ctx_stored_as_json_segments(self, logs_db):
        conn, cursor = logs_db
        database.unpack([log("loss", 0.5, ctx=[seg("epoch", 2)])], cursor)
        cursor.execute("SELECT ctx FROM logs")
        assert cursor.fetchone()[0] == (
            '[{"name": "epoch", "iteration": 2, "value": null}]'
        )

    def test_accepts_jsonl_dicts(self, logs_db):
        # `flor unpack` feeds raw dicts straight off disk.
        conn, cursor = logs_db
        record = {
            "projid": "proj",
            "tstamp": "2026-05-22T10:00:00.000000",
            "filename": "train.py",
            "ctx": [{"name": "epoch", "iteration": 1, "value": None}],
            "name": "loss",
            "value": 0.25,
            "type": 1,
        }
        database.unpack([record], cursor)
        cursor.execute("SELECT value_name, value FROM logs")
        assert cursor.fetchone() == ("loss", "0.25")


class TestCreateTables:
    def test_migrates_pre_source_schema(self):
        conn = sqlite3.connect(":memory:")
        cursor = conn.cursor()
        cursor.execute(
            "CREATE TABLE logs (projid TEXT, tstamp TEXT, filename TEXT, ctx TEXT, "
            "value_name TEXT, value TEXT, value_type INTEGER)"
        )
        cursor.execute("INSERT INTO logs VALUES ('p','t','f',NULL,'loss','0.5',1)")

        database.create_tables(cursor)

        cursor.execute("SELECT value_name, source FROM logs")
        # Pre-tag rows only ever came from JSONL, which is forward truth.
        assert cursor.fetchall() == [("loss", "forward")]
        conn.close()

    def test_drops_pre_ctx_schema(self):
        conn = sqlite3.connect(":memory:")
        cursor = conn.cursor()
        cursor.execute(
            "CREATE TABLE logs (projid TEXT, tstamp TEXT, ctx_id INTEGER, "
            "value_name TEXT, value TEXT)"
        )
        cursor.execute("CREATE TABLE loops (ctx_id INTEGER, name TEXT)")

        database.create_tables(cursor)

        cursor.execute("PRAGMA table_info(logs)")
        assert {row[1] for row in cursor.fetchall()} >= {"ctx", "source"}
        cursor.execute("SELECT name FROM sqlite_master WHERE name='loops'")
        assert cursor.fetchone() is None
        conn.close()

    def test_is_idempotent(self, logs_db):
        conn, cursor = logs_db
        database.unpack([log("loss", 0.5)], cursor)
        database.create_tables(cursor)
        cursor.execute("SELECT COUNT(*) FROM logs")
        assert cursor.fetchone()[0] == 1


class TestPivot:
    def test_surfaces_source_and_loop_columns(self, logs_db):
        conn, cursor = logs_db
        database.unpack(
            [
                log("loss", 0.5, ctx=[seg("epoch", 0)]),
                log("loss", 0.25, ctx=[seg("epoch", 1)]),
            ],
            cursor,
            source="forward",
        )
        conn.commit()

        df = database.pivot(conn, "loss")

        assert "source" in df.columns
        assert df["epoch"].tolist() == [0, 1]
        assert df["loss"].tolist() == ["0.5", "0.25"]

    def test_joins_keep_forward_and_replay_apart(self, logs_db):
        conn, cursor = logs_db
        database.unpack(
            [
                log("loss", 0.5, ctx=[seg("epoch", 0)]),
                log("val_acc", 90, ctx=[seg("epoch", 0)]),
            ],
            cursor,
            source="forward",
        )
        database.unpack(
            [log("val_acc", 91, ctx=[seg("epoch", 0)])], cursor, source="replay"
        )
        conn.commit()

        df = database.pivot(conn, "loss", "val_acc")

        forward = df[df["source"] == "forward"]
        replay = df[df["source"] == "replay"]
        # The replayed val_acc must not be paired with the forward loss.
        assert forward["val_acc"].tolist() == ["90"]
        assert replay["val_acc"].tolist() == ["91"]
        assert replay["loss"].isna().all()

    def test_pivot_star_uses_run_level_rows(self, logs_db):
        conn, cursor = logs_db
        database.unpack(
            [
                log("lr", 0.001),
                log("loss", 0.5, ctx=[seg("epoch", 0)]),
            ],
            cursor,
        )
        conn.commit()

        df = database.pivot(conn)

        assert "lr" in df.columns
        assert "loss" not in df.columns  # loop-scoped, not a run-level arg
        assert df["lr"].tolist() == ["0.001"]

    def test_nested_ctx_surfaces_every_depth(self, logs_db):
        conn, cursor = logs_db
        database.unpack(
            [log("loss", 0.5, ctx=[seg("epoch", 1), seg("step", 7)])], cursor
        )
        conn.commit()

        df = database.pivot(conn, "loss")

        assert df["epoch"].tolist() == [1]
        assert df["step"].tolist() == [7]

    def test_outer_loops_come_before_inner_ones(self, logs_db):
        conn, cursor = logs_db
        database.unpack(
            [log("loss", 0.5, ctx=[seg("epoch", 1), seg("step", 7)])], cursor
        )
        conn.commit()

        cols = list(database.pivot(conn, "loss").columns)

        # Nesting order, left to right: the loop that encloses reads first.
        assert cols.index("epoch") < cols.index("step")

    def test_value_column_is_dropped_when_it_repeats_the_index(self, logs_db):
        conn, cursor = logs_db
        database.unpack(
            [log("loss", 0.5, ctx=[seg("epoch", 0, "0"), seg("step", 7, "7")])], cursor
        )
        conn.commit()

        df = database.pivot(conn, "loss")

        # Looping over a range makes value and index the same column twice.
        assert "epoch_value" not in df.columns
        assert "step_value" not in df.columns
        assert df["epoch"].tolist() == [0]
        assert df["step"].tolist() == [7]

    def test_value_column_survives_when_it_differs_from_the_index(self, logs_db):
        conn, cursor = logs_db
        database.unpack(
            [
                log("loss", 0.5, ctx=[seg("epoch", 0, "0"), seg("step", 0, "a")]),
                log("loss", 0.4, ctx=[seg("epoch", 0, "0"), seg("step", 1, "b")]),
            ],
            cursor,
        )
        conn.commit()

        df = database.pivot(conn, "loss")

        assert "epoch_value" not in df.columns
        assert df["step_value"].tolist() == ["a", "b"]
