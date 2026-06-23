"""Tests for `assembly control` — actions, tools, the engine loop, the LLM
bridge, and rendering.

Every external leg is faked (see tests/_control_helpers.py), so the loop is
exercised with no microphone, network, subprocess, or macOS. Helper-transport,
listen, and command-wiring tests live in test_control_exec.py.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from aai_cli.control import actions, bridge, engine, prompt, render, tools
from aai_cli.control.actions import Action, InvalidAction
from tests._control_helpers import RecordingRenderer, fake_completion, last_json, scripted
from tests._snapshot_surface import normalize

# --- actions -----------------------------------------------------------------


def test_validate_returns_action_for_known_name_with_required_args():
    action = actions.validate("type_text", {"text": "hi"})
    assert action == Action(name="type_text", arguments={"text": "hi"})


def test_validate_rejects_unknown_action():
    with pytest.raises(InvalidAction, match="Unknown action 'bogus'"):
        actions.validate("bogus", {})


def test_validate_rejects_missing_required_argument():
    with pytest.raises(InvalidAction, match="missing required argument"):
        actions.validate("type_text", {})


def test_is_observe_only_true_for_read_only_actions():
    assert actions.validate("get_ui_tree", {}).is_observe() is True
    assert actions.validate("type_text", {"text": "x"}).is_observe() is False


def test_request_merges_action_name_and_arguments():
    request = Action(name="key_combo", arguments={"keys": ["cmd", "s"]}).request()
    assert request == {"action": "key_combo", "keys": ["cmd", "s"]}


# --- tools --------------------------------------------------------------------


def test_tool_names_match_executable_actions():
    assert set(tools.tool_names()) == set(actions.ACTION_SPECS)
    assert tools.tool_names() == tuple(sorted(actions.ACTION_SPECS))


def test_tool_definitions_carry_required_args_from_specs():
    # Round-trip through JSON so the nested schema is plain data to index into.
    entries = json.loads(json.dumps(tools.tool_definitions()))
    defs = {entry["function"]["name"]: entry for entry in entries}
    assert len(defs) == len(actions.ACTION_SPECS)
    assert defs["type_text"]["function"]["parameters"]["required"] == ["text"]
    assert defs["get_ui_tree"]["function"]["parameters"]["required"] == []
    assert defs["type_text"]["type"] == "function"
    # The schema forbids extra args, so a model can't smuggle unknown fields.
    assert defs["type_text"]["function"]["parameters"]["additionalProperties"] is False


# --- engine message shaping ---------------------------------------------------


def test_assistant_message_serializes_tool_calls():
    reply = engine.Reply(
        content="ok",
        tool_calls=(engine.ToolCall(id="c1", name="type_text", arguments={"text": "x"}),),
    )
    message = json.loads(json.dumps(engine._assistant_message(reply)))
    assert message["role"] == "assistant"
    assert message["content"] == "ok"  # truthy content is kept, not dropped to null
    call = message["tool_calls"][0]
    assert call["id"] == "c1"
    assert call["type"] == "function"
    assert call["function"]["name"] == "type_text"
    assert call["function"]["arguments"] == json.dumps({"text": "x"})


def test_assistant_message_without_tool_calls_has_no_tool_calls_key():
    message = json.loads(json.dumps(engine._assistant_message(engine.Reply("done", ()))))
    assert message == {"role": "assistant", "content": "done"}


def test_tool_message_carries_call_id_and_json_result():
    message = json.loads(json.dumps(engine._tool_message("c9", {"ok": True})))
    assert message == {"role": "tool", "tool_call_id": "c9", "content": json.dumps({"ok": True})}


# --- engine loop --------------------------------------------------------------


def test_run_turn_executes_tool_call_then_speaks_reply():
    renderer = RecordingRenderer()
    executed: list[Action] = []

    def execute(action: Action) -> dict[str, object]:
        executed.append(action)
        return {"ok": True}

    replies = [
        engine.Reply(content="", tool_calls=(engine.ToolCall("c1", "type_text", {"text": "hi"}),)),
        engine.Reply(content="typed it", tool_calls=()),
    ]
    engine.run_turn(
        "type hi",
        [{"role": "system", "content": "s"}],
        respond=scripted(replies),
        execute=execute,
        renderer=renderer,
        max_steps=5,
        allow_mutate=True,
    )
    assert renderer.users == ["type hi"]
    assert executed == [Action("type_text", {"text": "hi"})]
    assert renderer.results == [(Action("type_text", {"text": "hi"}), {"ok": True})]
    assert renderer.replies == ["typed it"]


def test_run_turn_dry_run_refuses_mutating_action_but_runs_observe():
    renderer = RecordingRenderer()
    executed: list[Action] = []

    def execute(action: Action) -> dict[str, object]:
        executed.append(action)
        return {"ok": True, "elements": []}

    replies = [
        engine.Reply(content="", tool_calls=(engine.ToolCall("c1", "type_text", {"text": "x"}),)),
        engine.Reply(content="", tool_calls=(engine.ToolCall("c2", "get_ui_tree", {}),)),
        engine.Reply(content="done", tool_calls=()),
    ]
    messages = engine.run_turn(
        "look",
        [],
        respond=scripted(replies),
        execute=execute,
        renderer=renderer,
        max_steps=5,
        allow_mutate=False,
    )
    # The mutating action was refused (never executed); the observe action ran.
    assert executed == [Action("get_ui_tree", {})]
    assert renderer.refused and renderer.refused[0][0].name == "type_text"
    # The refused tool call is reported back to the model as a failure (ok False).
    refused_msg = json.loads(json.dumps(messages[2]))
    assert json.loads(refused_msg["content"])["ok"] is False


def test_run_turn_reports_invalid_tool_call_without_executing():
    renderer = RecordingRenderer()
    executed: list[Action] = []

    replies = [
        engine.Reply(content="", tool_calls=(engine.ToolCall("c1", "bogus", {}),)),
        engine.Reply(content="sorry", tool_calls=()),
    ]
    messages = engine.run_turn(
        "do bad",
        [],
        respond=scripted(replies),
        execute=lambda action: executed.append(action) or {"ok": True},
        renderer=renderer,
        max_steps=5,
        allow_mutate=True,
    )
    assert executed == []
    assert renderer.invalid and "Unknown action" in renderer.invalid[0]
    # The invalid call is reported back to the model as a failure (ok False).
    invalid_msg = json.loads(json.dumps(messages[2]))
    assert json.loads(invalid_msg["content"])["ok"] is False


def test_run_turn_stops_at_step_limit_with_a_spoken_note():
    renderer = RecordingRenderer()
    # Always returns a tool call -> never settles -> must hit the step budget.
    forever = engine.Reply(content="", tool_calls=(engine.ToolCall("c", "get_ui_tree", {}),))
    engine.run_turn(
        "loop",
        [],
        respond=scripted([forever, forever]),
        execute=lambda action: {"ok": True},
        renderer=renderer,
        max_steps=2,
        allow_mutate=True,
    )
    assert renderer.replies == [engine.STEP_LIMIT_REPLY]


def test_run_session_threads_system_prompt_and_history_across_turns():
    renderer = RecordingRenderer()
    seen: list[list[dict[str, object]]] = []

    def respond(messages: list[engine.Message]) -> engine.Reply:
        seen.append([dict(m) for m in messages])
        return engine.Reply(content="ack", tool_calls=())

    engine.run_session(
        ["first", "second"],
        system="SYS",
        respond=respond,
        execute=lambda action: {"ok": True},
        renderer=renderer,
        max_steps=3,
        allow_mutate=True,
    )
    assert renderer.replies == ["ack", "ack"]
    # First call: system + first user. Second call also starts with the system prompt
    # and carries the first turn forward (history threading).
    assert seen[0][0] == {"role": "system", "content": "SYS"}
    assert seen[0][-1] == {"role": "user", "content": "first"}
    assert seen[1][0] == {"role": "system", "content": "SYS"}
    assert any(m.get("content") == "first" for m in seen[1])
    assert seen[1][-1] == {"role": "user", "content": "second"}


def test_system_prompt_is_nonempty_spoken_brief():
    assert "tools" in prompt.system_prompt()


# --- bridge (LLM Gateway adapter) ---------------------------------------------


def test_parse_arguments_handles_valid_empty_and_malformed():
    assert bridge._parse_arguments(json.dumps({"a": 1})) == {"a": 1}
    assert bridge._parse_arguments("") == {}
    assert bridge._parse_arguments("not json") == {}
    assert bridge._parse_arguments("[1, 2]") == {}


def test_reply_of_converts_message_and_tool_calls():
    call = SimpleNamespace(
        id="t1",
        type="function",
        function=SimpleNamespace(name="focus_app", arguments=json.dumps({"name": "Safari"})),
    )
    reply = bridge._reply_of(fake_completion("hello", [call]))
    assert reply.content == "hello"
    assert reply.tool_calls == (engine.ToolCall("t1", "focus_app", {"name": "Safari"}),)


def test_reply_of_skips_non_function_tool_calls():
    custom = SimpleNamespace(id="t2", type="custom", function=None)
    reply = bridge._reply_of(fake_completion("", [custom]))
    assert reply.tool_calls == ()


def test_reply_of_defaults_missing_content_and_tool_calls():
    reply = bridge._reply_of(fake_completion(None, None))
    assert reply.content == ""
    assert reply.tool_calls == ()


def test_build_responder_passes_tools_in_extra_and_returns_reply():
    seen = {}

    def fake_complete(api_key, *, model, messages, max_tokens, extra):
        seen.update(api_key=api_key, model=model, max_tokens=max_tokens, extra=extra)
        return fake_completion("ok", None)

    respond = bridge.build_responder("k", model="m", max_tokens=7, complete=fake_complete)
    reply = respond([{"role": "user", "content": "hi"}])
    assert reply == engine.Reply(content="ok", tool_calls=())
    assert seen["api_key"] == "k"
    assert seen["model"] == "m"
    assert seen["max_tokens"] == 7
    assert seen["extra"]["tool_choice"] == "auto"
    assert {t["function"]["name"] for t in seen["extra"]["tools"]} == set(actions.ACTION_SPECS)


# --- render -------------------------------------------------------------------


def test_describe_includes_arguments_only_when_present():
    assert render._describe(Action("get_ui_tree", {})) == "get_ui_tree"
    assert "Safari" in render._describe(Action("focus_app", {"name": "Safari"}))


def test_renderer_json_mode_emits_typed_events(capsys):
    r = render.ControlRenderer(json_mode=True)
    r.on_user("hello")
    assert last_json(capsys.readouterr().out) == {"type": "user", "text": "hello"}

    r.on_action(Action("focus_app", {"name": "Safari"}))
    event = last_json(capsys.readouterr().out)
    assert event == {"type": "action", "action": "focus_app", "arguments": {"name": "Safari"}}

    r.on_result(Action("focus_app", {"name": "Safari"}), {"ok": True})
    assert last_json(capsys.readouterr().out)["type"] == "result"

    r.on_refused(Action("type_text", {"text": "x"}), "nope")
    assert last_json(capsys.readouterr().out) == {
        "type": "refused",
        "action": "type_text",
        "reason": "nope",
    }

    r.on_invalid("bad call")
    assert last_json(capsys.readouterr().out) == {"type": "invalid", "reason": "bad call"}

    r.on_reply("all set")
    assert last_json(capsys.readouterr().out) == {"type": "reply", "text": "all set"}


def test_renderer_human_mode_splits_progress_and_reply(capsys):
    r = render.ControlRenderer(json_mode=False)
    r.on_user("hello")
    r.on_action(Action("focus_app", {"name": "Safari"}))
    r.on_refused(Action("type_text", {"text": "x"}), "nope")
    r.on_invalid("bad call")
    r.on_reply("all set")
    captured = capsys.readouterr()
    err = normalize(captured.err)
    out = normalize(captured.out)
    # Progress narration is on stderr; the spoken reply is the only thing on stdout.
    assert "hello" in err
    assert "focus_app" in err
    assert "nope" in err
    assert "bad call" in err
    assert out.strip() == "all set"
    # The reply line is bare text, not a JSON event (kills the json_mode mutant).
    with pytest.raises(json.JSONDecodeError):
        json.loads(out.strip())


def test_renderer_human_result_is_quiet_on_success_loud_on_failure(capsys):
    r = render.ControlRenderer(json_mode=False)
    r.on_result(Action("type_text", {"text": "x"}), {"ok": True})
    assert capsys.readouterr().err.strip() == ""
    r.on_result(Action("type_text", {"text": "x"}), {"ok": False, "error": "denied"})
    assert "denied" in normalize(capsys.readouterr().err)
