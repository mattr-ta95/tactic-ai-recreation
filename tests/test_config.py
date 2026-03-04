"""Tests for configuration utilities (validates Fix 1: no eval)."""

from utils.config import parse_cli_overrides, _safe_parse_value


class TestSafeParseValue:
    """Verify _safe_parse_value handles common types without eval()."""

    def test_int(self):
        assert _safe_parse_value("42") == 42

    def test_float(self):
        assert _safe_parse_value("0.001") == 0.001

    def test_true(self):
        assert _safe_parse_value("true") is True
        assert _safe_parse_value("True") is True
        assert _safe_parse_value("yes") is True

    def test_false(self):
        assert _safe_parse_value("false") is False
        assert _safe_parse_value("False") is False

    def test_none(self):
        assert _safe_parse_value("none") is None
        assert _safe_parse_value("null") is None

    def test_string_passthrough(self):
        assert _safe_parse_value("hello") == "hello"

    def test_list_literal(self):
        assert _safe_parse_value("[1, 2, 3]") == [1, 2, 3]


class TestParseCliOverrides:

    def test_nested_key(self):
        result = parse_cli_overrides(["model.hidden_dim=256"])
        assert result == {"model": {"hidden_dim": 256}}

    def test_bool_value(self):
        result = parse_cli_overrides(["training.use_augmentation=true"])
        assert result == {"training": {"use_augmentation": True}}

    def test_ignores_no_equals(self):
        result = parse_cli_overrides(["no_equals_sign"])
        assert result == {}
