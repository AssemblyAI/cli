"""Unit tests for the `assembly code` gateway model wiring (code_agent/model.py).

Split out of test_code_agent.py to stay under the 500-line file gate. These cover the
``_GatewayChatOpenAI`` subclass and its helpers that paper over the LLM Gateway's
OpenAI-incompatible quirks: content flattening, streamed tool-call id hoisting, dropping
the gateway's spurious blank tool-call deltas, and filling empty tool-call arguments.
"""

from __future__ import annotations

from aai_cli.code_agent import model as model_mod
from aai_cli.core import environments


def test_build_model_targets_the_gateway():  # untyped: probes ChatOpenAI subclass attrs
    m = model_mod.build_model("sk-test", model="claude-sonnet-4-6")
    assert m.model_name == "claude-sonnet-4-6"
    assert m.openai_api_base == environments.active().llm_gateway_base
    assert m.use_responses_api is False


def test_build_model_flattens_list_content_for_gateway():  # untyped: probes the payload dict
    from langchain_core.messages import HumanMessage, SystemMessage

    m = model_mod.build_model("sk-test", model="claude-sonnet-4-6")
    # deepagents hands the model multi-block content arrays; the gateway 500s on those,
    # so the model must flatten each message's content to a plain string before sending.
    payload = m._get_request_payload(
        [
            SystemMessage(content=[{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]),
            HumanMessage(content="hi"),
        ]
    )
    assert [msg["content"] for msg in payload["messages"]] == ["ab", "hi"]


def test_flatten_content_guards() -> None:
    model_mod._flatten_content(None)  # not a list -> early return, no error
    items = ["raw", 123]  # non-dict members are skipped, list left untouched
    model_mod._flatten_content(items)
    assert items == ["raw", 123]


def test_hoist_tool_call_ids_moves_id_out_of_function_only_when_missing() -> None:
    # One chunk exercising every branch: each malformed variant is skipped, and only a
    # tool call carrying a function-nested id gets hoisted. Hold references to the inner
    # dicts so the in-place mutation is asserted with a clean type.
    noid_fn: dict[str, object] = {"name": "b"}
    hoist_fn: dict[str, object] = {"id": "HOIST", "name": "c", "arguments": ""}
    noid_call: dict[str, object] = {"index": 1, "function": noid_fn}
    hoist_call: dict[str, object] = {"index": 2, "function": hoist_fn}
    tool_calls: list[object] = [
        None,  # tool_call not a dict -> skipped
        {"index": 0, "function": 7},  # function not a dict -> skipped
        noid_call,  # function has no id -> nothing to hoist
        hoist_call,  # the real gateway shape -> id hoisted out of function
    ]
    chunk: dict[str, object] = {
        "choices": [
            None,  # choice not a dict -> skipped
            {"delta": None},  # delta not a dict -> skipped
            {"delta": {"content": "hi"}},  # no tool_calls -> skipped
            {"delta": {"tool_calls": 99}},  # tool_calls not a list -> skipped
            {"delta": {"tool_calls": tool_calls}},
        ]
    }
    model_mod._hoist_tool_call_ids(chunk)
    assert "id" not in noid_call  # no id invented for a call that never had one
    assert noid_fn == {"name": "b"}  # left untouched
    assert hoist_call["id"] == "HOIST"  # hoisted to the top level where langchain reads it
    assert "id" not in hoist_fn  # and removed from function so it isn't duplicated


def test_hoist_tool_call_ids_guards() -> None:
    model_mod._hoist_tool_call_ids(None)  # not a dict -> early return, no error
    model_mod._hoist_tool_call_ids({"choices": 99})  # choices not a list -> early return


def test_is_blank_tool_call() -> None:
    assert model_mod._is_blank_tool_call({"function": {"id": "", "name": "", "arguments": ""}})
    assert model_mod._is_blank_tool_call({"function": {}})  # all fields absent
    assert not model_mod._is_blank_tool_call({"function": {"name": "x"}})  # has a name
    assert not model_mod._is_blank_tool_call({"function": {"id": "i"}})  # has an id
    assert not model_mod._is_blank_tool_call({"function": {"arguments": "a"}})  # has arguments
    assert not model_mod._is_blank_tool_call({"function": 7})  # function not a dict
    assert not model_mod._is_blank_tool_call(None)  # not a dict


def test_hoist_tool_call_ids_drops_spurious_blank_delta() -> None:
    # The gateway prefixes every streamed turn with an empty tool-call delta; it must be
    # dropped (else a pure-text turn dispatches a tool call with name="").
    real_fn: dict[str, object] = {"id": "toolu_X", "name": "get_weather", "arguments": ""}
    real_call: dict[str, object] = {"index": 0, "function": real_fn}
    delta: dict[str, object] = {
        "tool_calls": [
            {"index": 0, "function": {"id": "", "name": "", "arguments": ""}},  # spurious blank
            real_call,
        ]
    }
    chunk: dict[str, object] = {"choices": [{"delta": delta}]}
    model_mod._hoist_tool_call_ids(chunk)
    assert delta["tool_calls"] == [real_call]  # blank dropped, real call kept
    assert real_call["id"] == "toolu_X"  # and its id hoisted out of function
    assert "id" not in real_fn


def test_convert_chunk_drops_spurious_blank_tool_call() -> None:
    from langchain_core.messages import AIMessageChunk
    from langchain_openai import ChatOpenAI

    m = model_mod.build_model("sk-test", model="claude-sonnet-4-6")
    assert isinstance(m, ChatOpenAI)  # narrow to the subclass that overrides the converter
    # A pure-text turn's leading delta carries only the gateway's blank tool call — the
    # converted chunk must surface no tool call (else deepagents dispatches a nameless tool).
    chunk = {
        "choices": [
            {
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "tool_calls": [
                        {"index": 0, "function": {"id": "", "name": "", "arguments": ""}}
                    ],
                },
                "finish_reason": None,
            }
        ]
    }
    gen = m._convert_chunk_to_generation_chunk(chunk, AIMessageChunk, None)
    assert gen is not None
    msg = gen.message
    assert isinstance(msg, AIMessageChunk)
    assert msg.tool_call_chunks == []  # the phantom blank tool call is gone


def test_is_empty_arguments() -> None:
    assert model_mod._is_empty_arguments("")  # empty string
    assert model_mod._is_empty_arguments("   ")  # whitespace only
    assert model_mod._is_empty_arguments("{}")  # empty object
    assert model_mod._is_empty_arguments("{ }")  # empty object with whitespace
    assert not model_mod._is_empty_arguments('{"path": "/"}')  # real arguments
    assert not model_mod._is_empty_arguments("[]")  # non-dict JSON is not "empty args"
    assert not model_mod._is_empty_arguments("{bad json")  # unparseable -> leave alone
    assert not model_mod._is_empty_arguments(None)  # non-string -> leave alone
    assert not model_mod._is_empty_arguments({"already": "parsed"})  # non-string -> leave alone


def test_ensure_tool_call_arguments_fills_only_empty_calls() -> None:
    empty_fn: dict[str, object] = {"name": "ls", "arguments": "{}"}
    blank_fn: dict[str, object] = {"name": "ls", "arguments": "  "}
    full_fn: dict[str, object] = {"name": "ls", "arguments": '{"path": "/"}'}
    tool_calls: list[object] = [
        None,  # tool_call not a dict -> skipped
        {"id": "0", "function": 7},  # function not a dict -> skipped
        {"id": "1", "function": empty_fn},  # empty object -> filled
        {"id": "2", "function": blank_fn},  # whitespace -> filled
        {"id": "3", "function": full_fn},  # real args -> untouched
    ]
    messages: list[object] = [
        None,  # message not a dict -> skipped
        {"role": "user", "content": "hi"},  # no tool_calls -> skipped
        {"tool_calls": 99},  # tool_calls not a list -> skipped
        {"tool_calls": tool_calls},
    ]
    model_mod._ensure_tool_call_arguments(messages)
    assert empty_fn["arguments"] == model_mod._PLACEHOLDER_ARGUMENTS
    assert blank_fn["arguments"] == model_mod._PLACEHOLDER_ARGUMENTS
    assert full_fn["arguments"] == '{"path": "/"}'  # left untouched


def test_ensure_tool_call_arguments_guards() -> None:
    model_mod._ensure_tool_call_arguments(None)  # not a list -> early return, no error
    model_mod._ensure_tool_call_arguments([{"tool_calls": 99}])  # tool_calls not a list


def test_get_request_payload_fills_empty_tool_call_arguments() -> None:
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_openai import ChatOpenAI

    m = model_mod.build_model("sk-test", model="claude-sonnet-4-6")
    assert isinstance(m, ChatOpenAI)  # narrow to the subclass that overrides the payload hook
    # An assistant tool call with no arguments serializes to arguments="{}", which the gateway
    # rejects (missing tool_use.input); the payload must carry the placeholder instead.
    ai = AIMessage(content="", tool_calls=[{"name": "ls", "args": {}, "id": "t1"}])
    payload = m._get_request_payload([HumanMessage(content="hi"), ai])
    calls = payload["messages"][1]["tool_calls"]
    assert calls[0]["function"]["arguments"] == model_mod._PLACEHOLDER_ARGUMENTS


def test_convert_chunk_hoists_streamed_tool_call_id() -> None:
    from langchain_core.messages import AIMessageChunk
    from langchain_openai import ChatOpenAI

    m = model_mod.build_model("sk-test", model="claude-sonnet-4-6")
    assert isinstance(m, ChatOpenAI)  # narrow to the subclass that overrides the converter
    # The gateway streams the tool-call id nested inside `function`; the override must hoist
    # it so langchain's converted chunk carries the id (else the reply ToolMessage gets None).
    chunk = {
        "choices": [
            {
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "tool_calls": [
                        {"index": 0, "function": {"id": "toolu_X", "name": "get_weather"}}
                    ],
                },
                "finish_reason": None,
            }
        ]
    }
    gen = m._convert_chunk_to_generation_chunk(chunk, AIMessageChunk, None)
    assert gen is not None
    msg = gen.message
    assert isinstance(msg, AIMessageChunk)
    assert msg.tool_call_chunks[0]["id"] == "toolu_X"
