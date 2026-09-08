"""Automatic capture of print / logging output.

The feature is defined as much by what it must *not* record as by what it does:
flor echoes every flor.log value to the terminal itself, and a logging record
that reaches a StreamHandler also shows up on stderr. Either one recorded
naively means every value is stored twice. The end-to-end classes below pin
both.
"""

import dataclasses
import io as _io
import json
import logging
import os
import subprocess

import pytest

from flordb import capture


@pytest.fixture
def cap():
    """`capture` wired to a list sink, with module state restored afterward."""
    saved_config = dataclasses.replace(capture.config)
    saved_sink = capture._sink
    recorded = []
    capture._sink = lambda channel, text: recorded.append((channel, text))
    capture._records_emitted = 0
    capture._cap_notified = False
    try:
        yield recorded
    finally:
        capture.config.__dict__.update(saved_config.__dict__)
        capture._sink = saved_sink
        capture._records_emitted = 0
        capture._cap_notified = False


def tee(channel=capture.STDOUT):
    return capture._Tee(_io.StringIO(), channel)


class TestTee:
    def test_forwards_to_the_wrapped_stream(self, cap):
        t = tee()
        t.write("visible\n")
        assert t.wrapped.getvalue() == "visible\n"

    def test_one_record_per_newline(self, cap):
        t = tee()
        t.write("first\nsecond\n")
        assert cap == [("io::stdout", "first"), ("io::stdout", "second")]

    def test_a_single_print_spans_several_writes(self, cap):
        # print("a", "b") calls write() four times: "a", " ", "b", "\n".
        t = tee()
        for chunk in ("a", " ", "b", "\n"):
            t.write(chunk)
        assert cap == [("io::stdout", "a b")]

    def test_unterminated_output_is_not_recorded_until_drained(self, cap):
        t = tee()
        t.write("no newline yet")
        assert cap == []
        t.drain()
        assert cap == [("io::stdout", "no newline yet")]

    def test_blank_lines_are_dropped(self, cap):
        t = tee()
        t.write("\n\n   \n")
        assert cap == []

    def test_progress_bar_repaints_collapse_to_one_record(self, cap):
        # A tqdm-shaped stream: many \r repaints, one closing newline.
        t = tee(capture.STDERR)
        for pct in range(0, 101, 10):
            t.write(f"\r{pct}%|{'#' * (pct // 10)}")
        t.write("\n")
        assert cap == [("io::stderr", "100%|##########")]

    def test_repainting_buffer_stays_bounded(self, cap):
        t = tee(capture.STDERR)
        for _ in range(500):
            t.write("\r" + "x" * 200)
        assert len(t._buf) <= capture.config.max_line

    def test_long_lines_are_truncated(self, cap):
        capture.config.max_line = 32
        t = tee()
        t.write("y" * 200 + "\n")
        (_, line), = cap
        assert line.endswith(capture.TRUNCATION_MARKER)
        assert len(line) == 32 + len(capture.TRUNCATION_MARKER)

    def test_record_ceiling_stops_capture(self, cap):
        capture.config.max_records = 3
        t = tee()
        for i in range(10):
            t.write(f"line {i}\n")
        assert [line for _, line in cap] == ["line 0", "line 1", "line 2"]

    def test_disabled_config_records_nothing_but_still_prints(self, cap):
        capture.config.enabled = False
        t = tee()
        t.write("still visible\n")
        assert cap == []
        assert t.wrapped.getvalue() == "still visible\n"

    def test_a_broken_sink_never_costs_the_user_their_output(self, cap):
        def boom(channel, text):
            raise RuntimeError("sink is down")

        capture._sink = boom
        t = tee()
        t.write("must still appear\n")
        assert t.wrapped.getvalue() == "must still appear\n"


class TestMuted:
    def test_muted_writes_pass_through_unrecorded(self, cap):
        t = tee()
        with capture.muted():
            t.write("flor's own message\n")
        assert cap == []
        assert t.wrapped.getvalue() == "flor's own message\n"

    def test_mute_is_restored_on_exit(self, cap):
        t = tee()
        with capture.muted():
            pass
        t.write("recorded\n")
        assert cap == [("io::stdout", "recorded")]

    def test_nesting_does_not_unmute_early(self, cap):
        t = tee()
        with capture.muted():
            with capture.muted():
                pass
            t.write("inner\n")
        assert cap == []


class TestLoggingDispatch:
    """`logging` is wrapped at Logger.handle, not registered as a Handler.

    A Handler would sit in handler order next to the user's StreamHandler,
    whose stderr write the tee would then record as a second row for the same
    message.
    """

    def dispatch(self, logger, level, msg, *args):
        record = logger.makeRecord(
            logger.name, level, __file__, 1, msg, args, None
        )
        capture._orig_handle = lambda _logger, _record: None
        try:
            capture._flor_handle(logger, record)
        finally:
            capture._orig_handle = None

    def test_level_names_the_channel(self, cap):
        logger = logging.getLogger("root")
        logger.name = "root"
        self.dispatch(logger, logging.WARNING, "careful")
        assert cap == [("io::log::warning", "careful")]

    def test_args_are_interpolated(self, cap):
        self.dispatch(logging.getLogger("root"), logging.INFO, "epoch %d", 3)
        assert cap == [("io::log::info", "epoch 3")]

    def test_non_root_logger_name_rides_along(self, cap):
        self.dispatch(logging.getLogger("trainer.val"), logging.ERROR, "diverged")
        assert cap == [("io::log::error", "trainer.val: diverged")]

    def test_multiline_messages_become_multiple_records(self, cap):
        self.dispatch(logging.getLogger("root"), logging.INFO, "one\ntwo")
        assert cap == [("io::log::info", "one"), ("io::log::info", "two")]

    def test_downstream_handlers_run_muted(self, cap):
        # The user's StreamHandler writes to the teed stderr while our wrapper
        # is dispatching; that write must not become a second record.
        seen = []

        def orig_handle(_logger, record):
            seen.append(capture._muted())

        logger = logging.getLogger("root")
        record = logger.makeRecord(
            "root", logging.INFO, __file__, 1, "hello", (), None
        )
        capture._orig_handle = orig_handle
        try:
            capture._flor_handle(logger, record)
        finally:
            capture._orig_handle = None
        assert seen == [True]
        assert cap == [("io::log::info", "hello")]


class TestExtractPairs:
    @pytest.mark.parametrize(
        "line,expected",
        [
            ("loss: 0.42", [("loss", 0.42)]),
            ("acc=0.91", [("acc", 0.91)]),
            ("train loss: 0.5, val loss: 0.7", [("loss", 0.5)]),
            ("lr: 1e-3", [("lr", 0.001)]),
            ("delta: -3.5", [("delta", -3.5)]),
            ("val_acc : 90", [("val_acc", 90.0)]),
            ("epoch 1 | loss: 0.4 | acc: 0.9", [("loss", 0.4), ("acc", 0.9)]),
        ],
    )
    def test_recognized(self, line, expected):
        assert capture.extract_pairs(line) == expected

    @pytest.mark.parametrize(
        "line",
        [
            "fetching http://host:80/data",
            "Epoch 1/10",
            "started at 2026-08-13T11:27:06",
            "elapsed 11:27:06",
            "acc: 90%",
            "checkpoint=/tmp/ckpt.pth",
            "device: cuda",
            "no pairs here at all",
        ],
    )
    def test_rejected(self, line):
        assert capture.extract_pairs(line) == []

    def test_first_occurrence_of_a_key_wins(self):
        assert capture.extract_pairs("loss: 0.5 loss: 0.9") == [("loss", 0.5)]

    def test_pairs_per_line_are_capped(self):
        line = " ".join(f"m{i}: {i}" for i in range(50))
        assert len(capture.extract_pairs(line)) == capture.MAX_PAIRS_PER_LINE


class TestExtractFields:
    """The index/measure split. `epoch` addresses a row; `loss` fills a cell."""

    @pytest.mark.parametrize(
        "line,index",
        [
            ("epoch 0 | loss: 0.5", ("epoch", 0)),
            ("Epoch 1/10 loss: 0.4", ("epoch", 1)),
            ("step: 1200 | lr: 1e-3", ("step", 1200)),
            ("step=7 loss: 0.1", ("step", 7)),
            ("iteration 7 | val_loss=0.22", ("iteration", 7)),
            ("batch 32 | loss: 0.1", ("batch", 32)),
            # Not leading, but the key still names a step carrying a whole
            # number, so it classifies the same way.
            ("loss: 0.4, epoch: 3", ("epoch", 3)),
        ],
    )
    def test_step_prefixes_become_the_index(self, line, index):
        assert capture.extract_fields(line).index == index

    @pytest.mark.parametrize(
        "line",
        [
            "Loading 5 files",      # not a step name -- would invent a loop
            "epoch 1.5 | loss: 0.2",  # an index is a whole number
            "loss: 0.42",             # no index at all
            "trained for 3 epochs",   # the step name does not lead
            "elapsed 11:27:06",
        ],
    )
    def test_no_index_is_invented(self, line):
        assert capture.extract_fields(line).index is None

    def test_the_index_is_not_also_a_measure(self):
        fields = capture.extract_fields("epoch: 2 | loss: 0.3")
        assert fields.index == ("epoch", 2)
        assert fields.measures == [("loss", 0.3)]

    def test_measures_survive_beside_the_index(self):
        fields = capture.extract_fields("epoch 1/10 loss: 0.4 acc: 0.9")
        assert fields.measures == [("loss", 0.4), ("acc", 0.9)]

    def test_a_second_step_name_stays_a_measure(self):
        fields = capture.extract_fields("epoch 1 | step: 40 | loss: 0.2")
        assert fields.index == ("epoch", 1)
        assert ("step", 40.0) in fields.measures

    def test_falsy_when_nothing_was_found(self):
        assert not capture.extract_fields("no pairs here at all")
        assert capture.extract_fields("loss: 1")


class TestHelpers:
    def test_is_io_channel(self):
        assert capture.is_io_channel("io::stdout")
        assert capture.is_io_channel("io::stderr")
        assert capture.is_io_channel("io::log::info")
        assert not capture.is_io_channel("val_acc")
        assert not capture.is_io_channel("time::iter")

    def test_raw_stream_unwraps_a_tee(self):
        underlying = _io.StringIO()
        t = capture._Tee(underlying, capture.STDOUT)
        assert capture.raw_stream(t) is underlying

    def test_raw_stream_passes_through_a_plain_stream(self):
        plain = _io.StringIO()
        assert capture.raw_stream(plain) is plain

    def test_env_disabled(self, monkeypatch):
        for value in ("0", "false", "NO"):
            monkeypatch.setenv("FLOR_CAPTURE", value)
            assert capture.env_disabled()
        for value in ("1", "true", ""):
            monkeypatch.setenv("FLOR_CAPTURE", value)
            assert not capture.env_disabled()


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------

TRAIN = '''
import logging

import flordb as flor

logging.basicConfig(level=logging.INFO, format="%(message)s")

print("setup done")
for epoch in flor.loop("epoch", range(3)):
    print(f"epoch {epoch} loss: {1.0 / (epoch + 1):.4f}")
    logging.info("validated epoch %d", epoch)
    flor.log("val_acc", 90 + epoch)
'''


def _git(cwd, *args):
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def rows(project, *, source="forward", path=None):
    """(value_name, value, ctx) triples from the run's JSONL."""
    return [
        (r["name"], r["value"], r["ctx"]) for r in project.records(path)
    ] if source == "forward" else _db_rows(project, source)


def _db_rows(project, source):
    conn = project.db()
    try:
        return [
            (name, value, json.loads(ctx) if ctx else None)
            for name, value, ctx in conn.execute(
                "SELECT value_name, value, ctx FROM logs WHERE source = ?",
                (source,),
            )
        ]
    finally:
        conn.close()


def channels(triples):
    return [(n, v) for n, v, _ in triples if n.startswith("io::")]


@pytest.mark.slow
class TestCapturedRun:
    @pytest.fixture
    def trained(self, project):
        project.write("train.py", TRAIN)
        project.run("train.py")
        return project

    def test_prints_are_recorded_with_loop_context(self, trained):
        got = [
            (v, c) for n, v, c in rows(trained) if n == "io::stdout"
        ]
        assert ("setup done", None) in got
        for epoch in range(3):
            line = f"epoch {epoch} loss: {1.0 / (epoch + 1):.4f}"
            assert (line, [{"name": "epoch", "iteration": epoch, "value": str(epoch)}]) in got

    def test_captured_io_gets_its_own_value_type(self, trained):
        types = {
            r["name"]: r["type"] for r in trained.records()
        }
        assert types["io::stdout"] == 2
        assert types["io::log::info"] == 2
        assert types["val_acc"] == 1

    def test_flor_log_echo_is_not_captured(self, trained):
        # flor.log prints "epoch: 0, val_acc: 90" itself. Recording that would
        # store every metric twice -- once structured, once as text.
        io_lines = [v for _, v in channels(rows(trained))]
        assert not [line for line in io_lines if "val_acc" in line]
        assert len([n for n, _, _ in rows(trained) if n == "val_acc"]) == 3

    def test_a_logging_call_produces_exactly_one_record(self, trained):
        # basicConfig installs a StreamHandler on stderr, which the tee also
        # sees. One message, one row.
        info = [v for n, v, _ in rows(trained) if n == "io::log::info"]
        assert info == ["validated epoch 0", "validated epoch 1", "validated epoch 2"]
        stderr_rows = [v for n, v, _ in rows(trained) if n == "io::stderr"]
        assert not [line for line in stderr_rows if "validated epoch" in line]

    def test_flor_own_chatter_is_not_captured(self, trained):
        io_lines = [v for _, v in channels(rows(trained))]
        for noise in ("Run committed successfully", "Created and switched"):
            assert not [line for line in io_lines if noise in line]

    def test_capture_does_not_disturb_the_metric_table(self, project):
        # The same script run with capture on and with capture off. Capture
        # adds rows to `logs`, but none of them may reach the pivot: same
        # columns, same values, either way.
        project.write("train.py", TRAIN)
        project.run("train.py")
        project.write("train2.py", TRAIN)
        project.run("train2.py", env={"FLOR_CAPTURE": "0"})

        out = project.run(
            "-c",
            "import flordb as flor; print(flor.dataframe('val_acc').to_csv(index=False))",
        ).stdout
        lines = [line for line in out.splitlines() if line.strip()]

        assert lines[0] == "projid,tstamp,filename,source,epoch,val_acc"
        captured = sorted(l.split(",", 4)[4] for l in lines[1:] if ",train.py," in l)
        uncaptured = sorted(l.split(",", 4)[4] for l in lines[1:] if ",train2.py," in l)
        assert captured == uncaptured == ["0,90", "1,91", "2,92"]

    def test_stdout_still_reaches_the_terminal(self, trained):
        proc = trained.run("train.py")
        assert "setup done" in proc.stdout


@pytest.mark.slow
class TestZeroCodeChanges:
    """`import flordb` and nothing else -- the case the feature exists for.

    Nothing here reaches log / arg / loop, so captured io is the only thing
    that can register the run. Without that, commit() is never reached from
    the atexit hook and the entire run is dropped on the floor.
    """

    PRINT_ONLY = (
        "import flordb as flor\n"
        "\n"
        "for epoch in range(3):\n"
        "    print(f'epoch {epoch} | loss: {1.0 / (epoch + 2):.4f}')\n"
    )

    @pytest.fixture
    def ran(self, project):
        project.write("train.py", self.PRINT_ONLY)
        project.run("train.py")
        return project

    def test_the_run_is_written_at_all(self, ran):
        assert len(ran.run_files()) == 1

    def test_prints_are_the_records(self, ran):
        assert [v for n, v, _ in rows(ran) if n == "io::stdout"] == [
            "epoch 0 | loss: 0.5000",
            "epoch 1 | loss: 0.3333",
            "epoch 2 | loss: 0.2500",
        ]

    def test_the_run_still_gets_its_auto_commit(self, ran):
        subjects = [m.strip().splitlines()[0] for m in ran.git_log() if m.strip()]
        assert any(s.startswith("FLOR::Auto-commit::") for s in subjects)

    def test_the_run_reaches_the_cache(self, ran):
        conn = ran.db()
        try:
            (count,) = conn.execute(
                "SELECT COUNT(*) FROM logs WHERE value_name = 'io::stdout'"
            ).fetchone()
        finally:
            conn.close()
        assert count == 3

    def test_a_silent_script_records_nothing(self, project):
        # No io, no flor calls: importing flordb must stay inert.
        project.write("quiet.py", "import flordb as flor\nx = 1 + 1\n")
        project.run("quiet.py")
        assert project.run_files() == []


@pytest.mark.slow
class TestAdHocInvocations:
    """`python -c` and the bare REPL are not runs.

    Since captured io now registers a run, leaving capture on for these would
    turn every ad-hoc query into a recorded run with its own auto-commit.
    """

    def test_dash_c_does_not_record_a_run(self, project):
        project.write("train.py", TRAIN)
        project.run("train.py")
        before = len(project.run_files())
        project.run("-c", "import flordb as flor; print(flor.io())")
        assert len(project.run_files()) == before

    def test_dash_c_does_not_add_a_commit(self, project):
        project.write("train.py", TRAIN)
        project.run("train.py")
        before = len(project.git_log())
        project.run("-c", "import flordb as flor; print('noise')")
        assert len(project.git_log()) == before

    def test_reading_back_does_not_pollute_the_io_table(self, project):
        project.write("train.py", TRAIN)
        project.run("train.py")
        before = len(channels(rows(project)))
        project.run(
            "-c", "import flordb as flor; print(flor.io().to_string())"
        )
        assert len(channels(rows(project))) == before


@pytest.mark.slow
class TestDisabling:
    def test_env_var_turns_capture_off(self, project):
        project.write("train.py", TRAIN)
        proc = project.run("train.py", env={"FLOR_CAPTURE": "0"})
        assert "setup done" in proc.stdout
        assert channels(rows(project)) == []

    def test_set_capture_false_turns_it_off(self, project):
        project.write(
            "train.py",
            "import flordb as flor\n"
            "flor.set_capture(False)\n"
            "print('invisible to flor')\n"
            "flor.log('x', 1)\n",
        )
        project.run("train.py")
        assert channels(rows(project)) == []
        assert [n for n, _, _ in rows(project) if n == "x"] == ["x"]

    def test_record_ceiling_is_settable(self, project):
        project.write(
            "train.py",
            "import flordb as flor\n"
            "flor.set_capture(max_records=5)\n"
            "for i in range(50):\n"
            "    print('line', i)\n"
            "flor.log('done', 1)\n",
        )
        project.run("train.py")
        assert len(channels(rows(project))) == 5

    def test_flor_subcommands_do_not_capture_themselves(self, project):
        project.write("train.py", TRAIN)
        project.run("train.py")
        before = len(channels(rows(project)))
        project.run("-m", "flordb", "dataframe", "val_acc")
        project.run("-m", "flordb", "unpack")
        assert len(channels(rows(project))) == before


@pytest.mark.slow
class TestProgressBars:
    def test_a_tqdm_bar_does_not_produce_a_row_per_repaint(self, project):
        project.write(
            "train.py",
            "import flordb as flor\n"
            "from tqdm import tqdm\n"
            "for i in flor.loop('epoch', range(2)):\n"
            "    for _ in tqdm(range(200)):\n"
            "        pass\n"
            "flor.log('done', 1)\n",
        )
        project.run("train.py")
        # 200 repaints per bar, two bars. Only the final rendering of each
        # survives the \\r collapse.
        assert len(channels(rows(project))) <= 4


UNNAMED = '''
import flordb as flor

for epoch in range(3):
    print(f"epoch {epoch} | loss: {1.0 / (epoch + 1):.4f}")
'''


@pytest.mark.slow
class TestExtraction:
    """Extraction is a read-time derivation, driven from the CLI.

    Nothing here re-runs a script to change what was extracted: the point of
    moving it off the write path is that the text is already on disk.
    """

    def test_a_run_writes_no_extractions_on_its_own(self, project):
        project.write("train.py", TRAIN)
        project.run("train.py")
        assert [n for n, _, _ in rows(project) if n == "loss"] == []
        assert rows(project, source="extract") == []

    def test_preview_writes_nothing(self, project):
        project.write("train.py", TRAIN)
        project.run("train.py")
        proc = project.run("-m", "flordb", "capture", "--preview")
        assert "loss" in proc.stdout
        assert "Nothing was written" in proc.stdout
        assert rows(project, source="extract") == []

    def test_extract_writes_the_metrics_preview_promised(self, project):
        project.write("train.py", TRAIN)
        project.run("train.py")
        preview = project.run("-m", "flordb", "capture", "--preview").stdout
        project.run("-m", "flordb", "capture", "--extract")
        loss = [(v, c) for n, v, c in rows(project, source="extract") if n == "loss"]
        assert [float(v) for v, _ in loss] == [1.0, 0.5, 0.3333]
        assert [c[0]["iteration"] for _, c in loss] == [0, 1, 2]
        assert preview.count("<-") == len(loss)

    def test_the_raw_line_survives_alongside_the_extraction(self, project):
        project.write("train.py", TRAIN)
        project.run("train.py")
        project.run("-m", "flordb", "capture", "--extract")
        stdout = [v for n, v, _ in rows(project) if n == "io::stdout"]
        assert "epoch 0 loss: 1.0000" in stdout

    def test_extracted_metrics_reach_the_dataframe(self, project):
        project.write("train.py", TRAIN)
        project.run("train.py")
        project.run("-m", "flordb", "capture", "--extract")
        out = project.run(
            "-c",
            "import flordb as flor; print(flor.dataframe('loss').to_csv(index=False))",
        ).stdout
        assert "loss" in out.splitlines()[0]
        assert "1.0" in out

    def test_an_unnamed_loop_gets_its_index_back_from_the_text(self, project):
        """The reason extraction splits index from measure.

        This script never calls `flor.loop`, so every line shares one ctx.
        Reading `epoch` as a metric would make it a peer of `loss` and pivot
        would return the 3x3 join; reading it as an index gives three rows.
        """
        project.write("train.py", UNNAMED)
        project.run("train.py")
        project.run("-m", "flordb", "capture", "--extract")
        loss = [(v, c) for n, v, c in rows(project, source="extract") if n == "loss"]
        assert [c[0]["name"] for _, c in loss] == ["epoch"] * 3
        assert [c[0]["iteration"] for _, c in loss] == [0, 1, 2]
        assert [n for n, _, _ in rows(project, source="extract")] == ["loss"] * 3

        out = project.run(
            "-c",
            "import flordb as flor; print(flor.dataframe('loss').to_csv(index=False))",
        ).stdout
        assert len(out.strip().splitlines()) == 4  # header plus one row per epoch

    def test_extracting_twice_leaves_what_extracting_once_leaves(self, project):
        project.write("train.py", TRAIN)
        project.run("train.py")
        project.run("-m", "flordb", "capture", "--extract")
        once = rows(project, source="extract")
        project.run("-m", "flordb", "capture", "--extract")
        assert rows(project, source="extract") == once

    def test_unpack_leaves_the_derivation_alone(self, project):
        project.write("train.py", TRAIN)
        project.run("train.py")
        project.run("-m", "flordb", "capture", "--extract")
        before = rows(project, source="extract")
        project.run("-m", "flordb", "unpack")
        assert rows(project, source="extract") == before

    def test_unpack_does_not_invent_extractions(self, project):
        project.write("train.py", TRAIN)
        project.run("train.py")
        project.run("-m", "flordb", "unpack")
        assert rows(project, source="extract") == []

    def test_the_text_is_tracked_and_the_reading_is_not(self, project):
        """The reading is a view over committed data, not data of its own."""
        project.write("train.py", TRAIN)
        project.run("train.py")
        project.run("-m", "flordb", "capture", "--extract")

        tracked = project.git_tracked_files()
        assert any(f.startswith(".flor/runs/") for f in tracked)
        assert not any(f.startswith(".flor/extracted/") for f in tracked)
        assert not os.path.exists(os.path.join(project.root, ".flor", "extracted"))

    def test_a_rebuilt_cache_loses_the_columns_until_re_extracted(self, project):
        """Deleting the db is what a teammate's fresh clone looks like.

        The io survives -- it is committed in runs/ -- but the derivation does
        not, and `--extract` is what brings the columns back.
        """
        project.write("train.py", TRAIN)
        project.run("train.py")
        project.run("-m", "flordb", "capture", "--extract")
        before = rows(project, source="extract")
        assert before

        projid = os.path.basename(project.root)
        os.remove(os.path.join(project.root, ".flor", f"{projid}.db"))
        project.run("-m", "flordb", "unpack")
        assert rows(project, source="extract") == []

        project.run("-m", "flordb", "capture", "--extract")
        assert rows(project, source="extract") == before

    def test_extraction_commits_nothing(self, project):
        """No auto-commit, on any branch: there is no artifact to save."""
        project.write("train.py", TRAIN)
        project.run("train.py")
        _git(project.root, "checkout", "-q", "main")
        head = _git(project.root, "rev-parse", "HEAD")

        proc = project.run("-m", "flordb", "capture", "--extract")

        assert "not committed" in proc.stdout
        assert _git(project.root, "rev-parse", "HEAD") == head

    def test_set_capture_extract_says_where_extraction_went(self, project):
        project.write(
            "train.py",
            "import flordb as flor\nflor.set_capture(extract=True)\nprint('loss: 0.5')\n",
        )
        proc = project.run("train.py")
        assert "no longer has an effect" in proc.stdout
        assert "capture --extract" in proc.stdout
        assert rows(project, source="extract") == []


@pytest.mark.slow
class TestReplay:
    @pytest.fixture
    def trained(self, project):
        project.write("train.py", TRAIN)
        project.run("train.py")
        return project

    def test_a_projected_replay_stays_quiet(self, trained):
        trained.run(
            "train.py", "--replay_flor", "--apply", "val_acc", "--iter", "epoch=all"
        )
        assert channels(rows(trained, source="replay")) == []

    def test_io_can_be_projected_explicitly(self, trained):
        trained.run(
            "train.py",
            "--replay_flor",
            "--apply",
            "val_acc,io::stdout",
            "--iter",
            "epoch=all",
        )
        replayed = channels(rows(trained, source="replay"))
        assert [v for _, v in replayed] == [
            "setup done",
            "epoch 0 loss: 1.0000",
            "epoch 1 loss: 0.5000",
            "epoch 2 loss: 0.3333",
        ]

    def test_replay_does_not_touch_the_forward_jsonl(self, trained):
        before = trained.records()
        trained.run(
            "train.py",
            "--replay_flor",
            "--apply",
            "io::stdout",
            "--iter",
            "epoch=all",
        )
        assert trained.records() == before

    def test_unpack_wipes_replayed_io(self, trained):
        trained.run(
            "train.py", "--replay_flor", "--apply", "io::stdout", "--iter", "epoch=all"
        )
        assert channels(rows(trained, source="replay"))
        trained.run("-m", "flordb", "unpack")
        assert rows(trained, source="replay") == []
