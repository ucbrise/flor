import os
import sqlite3

import pytest

from flordb import database, orm


@pytest.fixture(autouse=True)
def extract_dir(tmp_path, monkeypatch):
    """Point `.flor/extracted/` at a scratch directory.

    write_extractions deletes the files it did not write this pass, so
    tests sharing one directory would delete each other's fixtures.
    """
    path = tmp_path / "extracted"
    path.mkdir()
    monkeypatch.setattr(orm, "EXTRACT_DIR", str(path))
    return str(path)


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


def io(line, ctx=None, channel="io::stdout", tstamp="2026-05-22T10:00:00.000000"):
    return orm.Log("proj", tstamp, "train.py", ctx, channel, line, 2)


class TestDeriveExtractions:
    def test_the_index_goes_to_ctx_and_the_measure_to_the_value(self, logs_db):
        conn, cursor = logs_db
        database.unpack([io("epoch 0 | loss: 0.5")], cursor)

        (derived, line), = database.derive_extractions(cursor)

        assert (derived.name, derived.value) == ("loss", 0.5)
        assert derived.ctx == [{"name": "epoch", "iteration": 0, "value": None}]
        assert line == "epoch 0 | loss: 0.5"

    def test_a_synthesized_index_makes_pivot_return_one_row_per_step(self, logs_db):
        """The failure the index/measure split exists to prevent.

        Three lines with no `flor.loop` share one ctx. As sibling metrics,
        `epoch` and `loss` would join on nothing and pivot would return 3x3.
        """
        conn, cursor = logs_db
        database.unpack(
            [io(f"epoch {e} | loss: {v}") for e, v in [(0, 0.5), (1, 0.3), (2, 0.2)]],
            cursor,
        )
        database.extract_metrics(cursor)
        conn.commit()

        df = database.pivot(conn, "loss")

        assert df["epoch"].tolist() == [0, 1, 2]
        assert df["loss"].tolist() == ["0.5", "0.3", "0.2"]

    def test_a_named_loop_keeps_its_own_index(self, logs_db):
        conn, cursor = logs_db
        database.unpack([io("epoch 0 | loss: 0.5", ctx=[seg("epoch", 0, "0")])], cursor)

        (derived, _), = database.derive_extractions(cursor)

        # Promoting the text would put `epoch` at two depths and split the
        # column; the recorded loop wins.
        assert [s["name"] for s in derived.ctx] == ["epoch"]

    def test_the_synthesized_index_nests_under_the_named_loop(self, logs_db):
        conn, cursor = logs_db
        database.unpack([io("step 4 | loss: 0.5", ctx=[seg("epoch", 2, "2")])], cursor)

        (derived, _), = database.derive_extractions(cursor)

        assert [(s["name"], s["iteration"]) for s in derived.ctx] == [
            ("epoch", 2),
            ("step", 4),
        ]

    def test_a_measure_never_shadows_a_recorded_loop(self, logs_db):
        conn, cursor = logs_db
        database.unpack([io("epoch: 1.5", ctx=[seg("epoch", 0, "0")])], cursor)

        assert database.derive_extractions(cursor) == []

    def test_the_last_line_wins_at_one_ctx(self, logs_db):
        conn, cursor = logs_db
        database.unpack(
            [io("epoch 0 | loss: 0.9"), io("epoch 0 | loss: 0.4")], cursor
        )

        (derived, _), = database.derive_extractions(cursor)

        assert derived.value == 0.4

    def test_only_io_rows_are_read(self, logs_db):
        conn, cursor = logs_db
        database.unpack([log("note", "epoch 0 | loss: 0.5")], cursor)

        assert database.derive_extractions(cursor) == []

    def test_every_channel_is_read(self, logs_db):
        """Extraction is a total function of the captured text.

        What gets committed must not depend on which subset of the run someone
        happened to ask about, so there is no way to narrow this.
        """
        conn, cursor = logs_db
        database.unpack(
            [
                io("epoch 0 | loss: 0.5"),
                io("epoch 0 | acc: 0.9", channel="io::log::info"),
                io("epoch 0 | grad: 1.5", channel="io::stderr"),
            ],
            cursor,
        )

        derived = database.derive_extractions(cursor)

        assert sorted(d.name for d, _ in derived) == ["acc", "grad", "loss"]


class TestExtractMetrics:
    def test_rows_are_tagged_derived(self, logs_db):
        conn, cursor = logs_db
        database.unpack([io("epoch 0 | loss: 0.5")], cursor)
        database.extract_metrics(cursor)

        cursor.execute("SELECT source FROM logs WHERE value_name = 'loss'")
        assert cursor.fetchall() == [("extract",)]

    def test_extracting_twice_does_not_double_the_rows(self, logs_db):
        conn, cursor = logs_db
        database.unpack([io("epoch 0 | loss: 0.5")], cursor)
        database.extract_metrics(cursor)
        database.extract_metrics(cursor)

        cursor.execute("SELECT COUNT(*) FROM logs WHERE source = 'extract'")
        assert cursor.fetchone() == (1,)

    def test_a_second_pass_drops_what_the_text_no_longer_says(self, logs_db):
        conn, cursor = logs_db
        database.unpack([io("epoch 0 | loss: 0.5")], cursor)
        database.extract_metrics(cursor)
        cursor.execute(f"DELETE FROM logs WHERE value_type = 2")
        database.unpack([io("epoch 0 | acc: 0.9")], cursor)

        database.extract_metrics(cursor)

        cursor.execute("SELECT value_name FROM logs WHERE source = 'extract'")
        assert cursor.fetchall() == [("acc",)]

    def test_the_reading_is_saved_beside_the_run(self, logs_db, extract_dir):
        conn, cursor = logs_db
        database.unpack([io("epoch 0 | loss: 0.5")], cursor)
        database.extract_metrics(cursor)

        (path,) = orm.extract_jsonl_paths()
        assert os.path.basename(path) == "2026-05-22T10:00:00.000000.jsonl"
        (record,) = orm.read_jsonl(path)
        assert record["name"] == "loss"
        assert record["ctx"] == [{"name": "epoch", "iteration": 0, "value": None}]

    def test_a_run_outside_this_pass_keeps_its_file(self, logs_db, extract_dir):
        """The files are committed, so absence must not mean deletion.

        A teammate's run can be in git -- its extraction file cloned along with
        it -- while this cache has never unpacked its io. Extraction must not
        read that silence as "no longer yields" and delete their reading.
        """
        conn, cursor = logs_db
        database.unpack([io("epoch 0 | loss: 0.5", tstamp="theirs")], cursor)
        database.extract_metrics(cursor)
        assert os.path.exists(orm.extract_jsonl_path("theirs"))

        # Their io leaves this cache; their committed file must not follow.
        cursor.execute("DELETE FROM logs")
        database.unpack([io("epoch 0 | acc: 0.9", tstamp="mine")], cursor)
        database.extract_metrics(cursor)

        assert os.path.exists(orm.extract_jsonl_path("theirs"))
        assert os.path.exists(orm.extract_jsonl_path("mine"))

    def test_a_run_that_stops_yielding_loses_its_file(self, logs_db, extract_dir):
        conn, cursor = logs_db
        database.unpack([io("epoch 0 | loss: 0.5")], cursor)
        database.extract_metrics(cursor)
        assert orm.extract_jsonl_paths()

        cursor.execute("DELETE FROM logs WHERE value_type = 2")
        database.unpack([io("nothing numeric here")], cursor)
        database.extract_metrics(cursor)

        assert orm.extract_jsonl_paths() == []

    def test_load_extractions_rebuilds_the_cache_from_the_files(
        self, logs_db, extract_dir
    ):
        conn, cursor = logs_db
        database.unpack([io("epoch 0 | loss: 0.5")], cursor)
        database.extract_metrics(cursor)

        # A fresh cache: the io is gone too, so nothing could be re-derived.
        cursor.execute("DELETE FROM logs")
        assert database.load_extractions(cursor) == 1

        cursor.execute("SELECT value_name, value, source FROM logs")
        assert cursor.fetchall() == [("loss", "0.5", "extract")]

    def test_load_extractions_replaces_rather_than_appends(
        self, logs_db, extract_dir
    ):
        conn, cursor = logs_db
        database.unpack([io("epoch 0 | loss: 0.5")], cursor)
        database.extract_metrics(cursor)
        database.load_extractions(cursor)
        database.load_extractions(cursor)

        cursor.execute("SELECT COUNT(*) FROM logs WHERE source = 'extract'")
        assert cursor.fetchone() == (1,)

    def test_derived_rows_stay_apart_from_forward_ones(self, logs_db):
        conn, cursor = logs_db
        database.unpack([io("epoch 0 | loss: 0.5"), log("val_acc", 90)], cursor)
        database.extract_metrics(cursor)
        conn.commit()

        df = database.pivot(conn, "loss")

        # `source` rides along, so a derived guess is never mistaken for a
        # value the script actually logged.
        assert df["source"].tolist() == ["extract"]
