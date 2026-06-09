from tests._cli_tree import leaf_command_items


def test_every_leaf_command_has_examples_epilog():
    missing = []
    for path, cmd in leaf_command_items():
        epilog = getattr(cmd, "epilog", None)
        if not (epilog and "Examples" in epilog):
            missing.append(" ".join(path))
    assert not missing, f"commands missing --help examples: {missing}"
