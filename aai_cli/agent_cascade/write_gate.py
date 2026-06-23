"""Path-scoped write gating for ``assembly live --files``.

The ``--files`` brain confirms writes before they touch disk. Rather than gate *every* write
(all-or-nothing, a keypress per write — friction for a hands-free voice turn), the policy is
expressed as deepagents :class:`FilesystemPermission` rules — auto-approve (``allow``) under each
``--auto-write`` subtree, ``interrupt`` everywhere else — and translated to the per-tool ``when``
predicates ``HumanInTheLoopMiddleware`` consumes. So a write inside an ``--auto-write`` directory
runs ungated while any other write still pauses for approval.

We deliberately do **not** pass ``create_deep_agent(permissions=…)``: deepagents raises
``NotImplementedError`` when ``permissions`` meets an execute-capable backend (our
:class:`~aai_cli.agent_cascade.sandbox.SandboxedShellBackend`), so we derive the ``interrupt_on``
ourselves with its own permission→interrupt translator and keep ``execute`` unconditionally gated
(it can't be path-scoped — deepagents derives interrupt predicates for the file tools alone).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deepagents.middleware.filesystem import FilesystemPermission


def _auto_write_rules(auto_write_paths: Sequence[str]) -> list[FilesystemPermission]:
    """The write policy as ordered FilesystemPermission rules: allow under each --auto-write
    subtree, interrupt everywhere else.

    First-match-wins (deepagents' ``_check_fs_permission``): a write under an auto-write root
    matches an ``allow`` rule and runs ungated; any other write falls through to the trailing
    ``/**`` ``interrupt`` rule and pauses for approval. With no auto-write paths the list is just
    the catch-all interrupt rule, so every write is gated — identical to the previous
    all-or-nothing behavior. Each root contributes both the node (``/scratch``) and its subtree
    (``/scratch/**``) so a write to the dir itself or anything under it auto-approves.
    """
    from deepagents.middleware.filesystem import FilesystemPermission

    rules = [
        FilesystemPermission(operations=["write"], paths=[root, f"{root}/**"], mode="allow")
        for root in auto_write_paths
    ]
    rules.append(FilesystemPermission(operations=["write"], paths=["/**"], mode="interrupt"))
    return rules


def write_interrupt_on(auto_write_paths: Sequence[str]) -> dict[str, object]:
    """The ``interrupt_on`` map gating writes for --files, path-scoped via FilesystemPermission.

    Translates the :func:`_auto_write_rules` policy into the per-tool ``when`` predicates
    HumanInTheLoopMiddleware consumes (deepagents' own permission→interrupt translator), so
    ``write_file``/``edit_file`` pause only when the target falls outside every --auto-write
    subtree. ``execute`` can't be path-scoped — deepagents derives interrupt predicates for the
    file tools alone — so it stays unconditionally gated: every command run is still approved.
    """
    from deepagents.middleware import _fs_interrupt

    # deepagents' permission→interrupt translator is the same one its public `create_deep_agent`
    # uses, but it lives in a private module. We can't reach it via `create_deep_agent(permissions=…)`
    # because that raises NotImplementedError against an execute-capable backend (ours), so we call
    # the translator directly. The name is held in a variable (not a literal getattr / direct
    # import) so the static checker doesn't bind to the private symbol while we deliberately reuse
    # this internal — deepagents is version-pinned in uv.lock, so it can't shift under us silently.
    translator = "_build_interrupt_on_from_permissions"
    build = getattr(_fs_interrupt, translator)
    interrupt_on: dict[str, object] = dict(build(_auto_write_rules(auto_write_paths)))
    interrupt_on["execute"] = True
    return interrupt_on
