from typing import TypeVar, Sequence

from slist import Slist

A = TypeVar("A")


def cli_input_list(options: Sequence[A], start_text: str) -> A:
    while True:
        cli_input = input(start_text + f"\nOptions: {options}\n")
        list_options = Slist(options)
        found_idx = list_options.find_one_idx(lambda x: str(x) == cli_input)
        if found_idx is not None:
            print(f"Chosen {cli_input}")
            return list_options[found_idx]
        else:
            print(f"Invalid input {cli_input}")
