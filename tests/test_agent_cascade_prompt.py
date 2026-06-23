"""Tests for the live agent's system-prompt construction (aai_cli.agent_cascade.prompt).

Split out of test_agent_cascade_brain.py to keep each file within the 500-line gate. The
prompt is tailored to the bound tools so the model is only told about capabilities it has.
"""

from __future__ import annotations

from aai_cli.agent_cascade import datetime_tool, prompt, weather_tool, webpage_tool


class _NamedTool:
    """A stand-in tool exposing just the ``.name`` the prompt builder inspects."""

    def __init__(self, name: str):
        self.name = name


def test_system_prompt_advertises_web_search_when_present():
    text = prompt.build_system_prompt(
        "You are a pirate.", tools=[_NamedTool(prompt.WEB_SEARCH_TOOL_NAME)]
    )
    # The persona is preserved, and the guidance advertises the web-search capability the
    # present tool backs (the plain cascade persona never mentions tools).
    assert text.startswith("You are a pirate.")
    assert "search the web" in text


def test_system_prompt_omits_web_search_when_search_tool_absent():
    # Without the Firecrawl search tool the guidance must NOT promise web search — announcing
    # a missing tool makes the agent narrate "I'll search…" and then stall with no answer. A
    # non-search tool name must not falsely trigger the web-search capability.
    text = prompt.build_system_prompt("persona", tools=[_NamedTool("some_other_tool")])
    assert "search the web for current or unfamiliar facts" not in text


def test_system_prompt_tells_model_not_to_promise_tools_when_none():
    # No tools at all: the model must answer from its own knowledge and explicitly not
    # promise to search or look anything up (the bug that left replies never coming back).
    text = prompt.build_system_prompt("persona", tools=[])
    assert "search the web for current or unfamiliar facts" not in text
    assert "your own knowledge" in text
    assert "Never say" in text


def test_extra_capability_lists_sorted_tool_names():
    # MCP tools are advertised generically, by name, alphabetically.
    phrase = prompt._extra_capability([_NamedTool("zeta"), _NamedTool("alpha")])
    assert phrase == "use your connected tools (alpha, zeta)"


def test_extra_capability_is_none_without_extra_tools():
    assert prompt._extra_capability([]) is None


def test_system_prompt_advertises_mcp_extra_tools():
    # With MCP tools bound (but no built-in legs), the model must be told it HAS tools —
    # not handed the "no external tools" guidance — and the tools are named.
    text = prompt.build_system_prompt("persona", tools=[], extra_tools=[_NamedTool("get_time")])
    assert "your own knowledge" not in text
    assert "use your connected tools (get_time)" in text


def test_system_prompt_advertises_files_when_enabled():
    # With --files on, the model must be told it can read/write files in the working dir,
    # so it knows the capability is real (and the no-tools guidance must not apply).
    text = prompt.build_system_prompt("persona", tools=[], files=True)
    assert "read, write, and search files in your working directory" in text
    assert "your own knowledge" not in text


def test_system_prompt_advertises_code_execution_under_files():
    prompt_text = prompt.build_system_prompt("persona", tools=[], files=True)
    assert "run code to solve problems" in prompt_text


def test_system_prompt_omits_code_execution_without_files():
    prompt_text = prompt.build_system_prompt("persona", tools=[], files=False)
    assert "run code" not in prompt_text


def test_system_prompt_advertises_delegation_under_files():
    # --files binds the task tool (a subagent), so the prompt offers delegating to a helper.
    assert "delegate a bigger job to a helper" in prompt.build_system_prompt(
        "persona", tools=[], files=True
    )


def test_system_prompt_omits_delegation_without_files():
    assert "delegate a bigger job" not in prompt.build_system_prompt(
        "persona", tools=[], files=False
    )


def test_system_prompt_omits_files_when_disabled():
    # Default: no file capability advertised (the model shouldn't promise file access it lacks).
    text = prompt.build_system_prompt("persona", tools=[], files=False)
    assert "working directory" not in text


def test_system_prompt_reports_tool_outcomes_honestly_when_tools_present():
    # A spoken agent that narrates a success it never achieved is worse than one that admits
    # it couldn't — so whenever tools are bound the guidance must tell the model not to claim
    # an action happened until the tool returns.
    text = prompt.build_system_prompt("persona", tools=[_NamedTool(prompt.WEB_SEARCH_TOOL_NAME)])
    assert "until the tool actually returns" in text


def test_system_prompt_warns_before_irreversible_file_actions():
    # The --files capability can write files and run code, which speaking can't undo, so the
    # model must be told to confirm before destructive actions and not claim a change landed.
    text = prompt.build_system_prompt("persona", tools=[], files=True)
    assert "can't be undone" in text


def test_system_prompt_omits_file_safety_warning_without_files():
    # The irreversibility warning is only meaningful when the file tools are actually bound.
    text = prompt.build_system_prompt("persona", tools=[], files=False)
    assert "can't be undone" not in text


def test_join_clause_grammar():
    # One/two/three capability phrases each render with natural conjunctions.
    assert prompt._join_clause(["a"]) == "a"
    assert prompt._join_clause(["a", "b"]) == "a and b"
    assert prompt._join_clause(["a", "b", "c"]) == "a, b, and c"


def test_tool_capabilities_lists_web_search_then_weather_when_both_present():
    caps = prompt._tool_capabilities(
        [_NamedTool(prompt.WEB_SEARCH_TOOL_NAME), _NamedTool(weather_tool.WEATHER_TOOL_NAME)]
    )
    # Exact list pins BOTH phrases and their order, killing a drop/swap of either block.
    assert caps == [
        "search the web for current or unfamiliar facts",
        "tell someone the current weather and short forecast for a place",
    ]


def test_read_url_tool_advertised_in_system_prompt():
    text = prompt.build_system_prompt(
        "persona", tools=[_NamedTool(webpage_tool.READ_URL_TOOL_NAME)]
    )
    assert "read a web page or PDF" in text


def test_weather_tool_advertised_in_system_prompt():
    text = prompt.build_system_prompt("persona", tools=[_NamedTool(weather_tool.WEATHER_TOOL_NAME)])
    assert "current weather and short forecast" in text
    # And it isn't the no-tools fallback.
    assert "no external tools" not in text


def test_datetime_tool_advertised_in_system_prompt():
    text = prompt.build_system_prompt(
        "persona", tools=[_NamedTool(datetime_tool.DATETIME_TOOL_NAME)]
    )
    assert "current date and time" in text
