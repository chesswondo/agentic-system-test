"""LangSmith tracing is on only with a key (and not switched off)."""

from picos.observability import configure_langsmith


def test_no_key_forces_tracing_off():
    env = {"LANGSMITH_TRACING": "true"}
    assert configure_langsmith("auto", env) is False
    assert env["LANGSMITH_TRACING"] == "false"


def test_key_enables_tracing_unless_off():
    env = {"LANGSMITH_API_KEY": "k"}
    assert configure_langsmith("auto", env) is True
    assert env["LANGSMITH_TRACING"] == "true"
    env = {"LANGSMITH_API_KEY": "k", "LANGSMITH_TRACING": "true"}
    assert configure_langsmith("off", env) is False
    assert env["LANGSMITH_TRACING"] == "false"


def test_explicit_false_with_key_is_respected():
    env = {"LANGSMITH_API_KEY": "k", "LANGSMITH_TRACING": "false"}
    assert configure_langsmith("auto", env) is False
