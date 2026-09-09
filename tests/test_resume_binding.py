"""Runtime binding of locally recognized resume targets."""

import sys

from flordb import api, cli


class Target:
    def load_state_dict(self, state):
        pass


def test_binding_preserves_an_existing_tracer_and_releases_setup_frame(clean_flags):
    filename = "/tmp/flor-resume-binding-test.py"
    source = (
        "def prep():\n"
        "    model = Target()\n"
        "    optimizer = Target()\n"
        "    return model, optimizer\n"
    )
    spec = cli.ResumeSpec(
        "ckpt.pth", "state", [("model", "model"), ("optimizer", "optimizer")],
        scope_name="prep", scope_lineno=1, lineno=4,
    )
    clean_flags.resume_spec = spec
    scope = {"Target": Target}
    exec(compile(source, filename, "exec"), scope)
    events = []

    def prior_trace(frame, event, arg):
        if frame.f_code.co_filename == filename:
            events.append(event)
        return prior_trace

    original = sys.gettrace()
    sys.settrace(prior_trace)
    stop = None
    try:
        stop = api._install_resume_binding(spec, filename)
        model, optimizer = scope["prep"]()
        assert sys.gettrace() is prior_trace
        assert api._resolve_targets(spec) == {"model": model, "optimizer": optimizer}
        assert "call" in events
        assert "line" in events
        assert "return" in events
    finally:
        if stop:
            stop()
        sys.settrace(original)


def test_unbound_function_targets_never_resolve_against_another_scope(clean_flags):
    import pytest

    spec = cli.ResumeSpec(
        "ckpt.pth", "state", [("model", "model")], scope_name="prep", scope_lineno=1,
    )
    with pytest.raises(RuntimeError, match="not bound before the replay loop"):
        api._resolve_targets(spec)


def test_skipped_guard_in_a_helper_without_an_explicit_return(clean_flags):
    filename = "/tmp/flor-resume-binding-guard-test.py"
    source = (
        "def prep(model, optimizer):\n"
        "    if False:\n"
        "        model.load_state_dict({})\n"
    )
    spec = cli.ResumeSpec(
        "ckpt.pth", "state", [("model", "model"), ("optimizer", "optimizer")],
        scope_name="prep", scope_lineno=1, lineno=3,
    )
    clean_flags.resume_spec = spec
    scope = {}
    exec(compile(source, filename, "exec"), scope)
    model, optimizer = Target(), Target()
    original = sys.gettrace()
    stop = api._install_resume_binding(spec, filename)
    try:
        scope["prep"](model, optimizer)
        assert sys.gettrace() is original
        assert api._resolve_targets(spec) == {"model": model, "optimizer": optimizer}
    finally:
        stop()
