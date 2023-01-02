from typing import NoReturn


def should_not_happen(msg: str = "This should not happen") -> NoReturn:
    raise AssertionError(msg)
