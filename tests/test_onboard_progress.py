from __future__ import annotations

from aai_cli import config
from aai_cli.onboard import progress


def test_requests_made_starts_at_zero() -> None:
    assert config.get_requests_made("default") == 0


def test_requests_made_is_zero_for_existing_profile_with_no_count() -> None:
    # A profile can exist (e.g. created by login) before any request is recorded;
    # its count must read 0, not the `or 1` fallback a mutant would introduce.
    config.set_profile_env("default", "production")
    assert config.get_requests_made("default") == 0


def test_record_request_increments_and_persists() -> None:
    assert config.record_request("default") == 1
    assert config.record_request("default") == 2
    assert config.get_requests_made("default") == 2


def test_goal_is_100() -> None:
    assert progress.GOAL == 100


def test_milestone_message_fires_only_at_milestones() -> None:
    assert progress.milestone_message(1) is not None
    assert progress.milestone_message(10) is not None
    assert progress.milestone_message(50) is not None
    assert progress.milestone_message(100) is not None
    assert progress.milestone_message(2) is None
    assert progress.milestone_message(0) is None


def test_render_progress_mentions_count_goal_and_usage_pointer() -> None:
    rendered = progress.render_progress(7)
    assert "7" in rendered
    assert "100" in rendered
    assert "aai usage" in rendered
    milestone = progress.render_progress(1)
    assert "activated" in milestone
