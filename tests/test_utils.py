import pandas as pd
import pytest

from flordb import utils


class TestDuckCast:
    @pytest.mark.parametrize(
        "raw,expected", [("true", True), ("1", True), ("no", False), ("", False)]
    )
    def test_bool(self, raw, expected):
        assert utils.duck_cast(raw, False) is expected

    def test_int(self):
        assert utils.duck_cast("64", 32) == 64

    def test_float(self):
        assert utils.duck_cast("5e-4", 1e-3) == pytest.approx(5e-4)

    def test_str(self):
        assert utils.duck_cast("cpu", "cuda") == "cpu"

    def test_bool_checked_before_int(self):
        # bool is a subclass of int; getting this order wrong turns
        # --override flag=true into an int() crash.
        assert utils.duck_cast("true", True) is True

    def test_unsupported_type(self):
        with pytest.raises(TypeError):
            utils.duck_cast("x", {"a": 1})


class TestToFilename:
    def test_uses_loop_value_when_present(self):
        assert (
            str(utils.to_filename({"epoch": (3, "3")}, "ckpt", ".pth"))
            == "ckpt_epoch_3.pth"
        )

    def test_falls_back_to_index_when_value_unjsonable(self):
        assert (
            str(utils.to_filename({"epoch": (3, None)}, "ckpt", ".pth"))
            == "ckpt_epoch_3.pth"
        )

    def test_nested_layers_outermost_first(self):
        name = utils.to_filename({"epoch": (1, None), "step": (7, None)}, "m", ".pth")
        assert str(name) == "m_epoch_1_step_7.pth"

    def test_no_layers(self):
        assert str(utils.to_filename({}, "ckpt", ".pth")) == "ckpt.pth"


class TestMisc:
    def test_is_jsonable(self):
        assert utils.is_jsonable([1, "a"])
        assert not utils.is_jsonable(object())

    def test_to_string_with_and_without_layers(self):
        assert utils.to_string({}, "loss", 0.5) == "loss: 0.5"
        assert utils.to_string({"epoch": (1, None)}, "loss", 0.5) == "epoch: 1, loss: 0.5"

    def test_cast_dtypes_numeric(self):
        df = pd.DataFrame({"loss": ["0.5", "0.25"], "note": ["a", "b"]})
        out = utils.cast_dtypes(df)
        assert out["loss"].dtype.kind == "f"
        assert out["note"].dtype == object or str(out["note"].dtype) == "str"

    def test_latest_keeps_only_max_tstamp(self):
        df = pd.DataFrame({"tstamp": ["2026-01-01", "2026-01-02"], "v": [1, 2]})
        assert utils.latest(df)["v"].tolist() == [2]

    @pytest.mark.parametrize(
        "seconds,expected",
        [
            (1, "under 10 seconds"),
            (50, "under 2 minutes"),
            (500, "up to 15 minutes"),
            (5000, "more than 15 minutes"),
        ],
    )
    def test_discretize(self, seconds, expected):
        assert utils.discretize(seconds) == expected
