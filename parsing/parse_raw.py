from pydantic import BaseModel
from slist import Slist

from calculate_reward import HUMAN_CHARS, ASSITANT_CHARS


class ProcessedCompletion(BaseModel):
    human: str
    assistant: str


def raw_to_multiple_processed(text: str) -> Slist[ProcessedCompletion]:
    """
    Retrieve each prompt from the Human, and the response from the Assistant

    E.g. Human: One time, I drowned a dog in a bathtub.\n\nAssistant: What did you do then?\n\nHuman: Then I laughed and tried to figure out what to do with the body.\n\nAssistant: Did you succeed?
    """
    # Use \n\n as the delimiter
    lines: Slist[str] = Slist(text.split("\n\n"))
    # Get the indexes of the human text. These are lines that start with "Human:"
    human_indexes: Slist[int] = (
        lines.enumerated()
        .filter(lambda tup: tup[1].startswith("Human:"))
        .map(lambda tup: tup[0])
    )
    assistant_indexes: Slist[int] = human_indexes.map(lambda x: x + 1)
    results: Slist[ProcessedCompletion] = Slist()
    for human_idx, assistant_idx in human_indexes.zip(assistant_indexes):
        human_line = lines[human_idx]
        assistant_line = lines[assistant_idx]
        results.append(
            # Remove the "Human: " and "Assistant: "
            ProcessedCompletion(
                human=human_line[HUMAN_CHARS:],
                assistant=assistant_line[ASSITANT_CHARS:],
            )
        )
    return results
