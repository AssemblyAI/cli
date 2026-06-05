from tests._cli_tree import leaf_command_items

# `version` is a trivial command with no flags; examples would be noise.
_EXEMPT = {"version"}


def test_every_leaf_command_has_examples_epilog():
    missing = []
    for path, cmd in leaf_command_items():
        name = path[-1] if path else cmd.name
        if name in _EXEMPT:
            continue
        epilog = getattr(cmd, "epilog", None)
        if not (epilog and "Examples" in epilog):
            missing.append(" ".join(path))
    assert not missing, f"commands missing --help examples: {missing}"
