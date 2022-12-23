from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pydantic import BaseModel
from slist import Slist

from api.json_operations import write_jsonl_file_from_basemodel
from api.moderations import OpenAIModeration, get_moderations_retry
from api.dataset_paths import anthropic_harmless_train_path, moderated_harmless_rejected
from parsing.parse_raw import ProcessedCompletion, raw_to_multiple_processed, raw_to_single_processed


class AnthropicRawFormat(BaseModel):
    chosen: str
    rejected: str

    class Config:
        frozen = True


class ProcessedWithModeration(BaseModel):
    processed: ProcessedCompletion
    moderation: OpenAIModeration


def get_raw_anthropic(path: Path) -> Slist[AnthropicRawFormat]:
    # read from anthropic_helpful_path
    jsonl_path = path
    # read the jsonl file
    with open(jsonl_path, "r") as f:
        lines = f.readlines()
        # create a list of ProcessedCompletion objects
        processed_completions = Slist[AnthropicRawFormat]()
        for line in lines:
            processed_completions.append(AnthropicRawFormat.parse_raw(line))
    return processed_completions


threadpool = ThreadPoolExecutor(max_workers=100)

if __name__ == "__main__":
    raw: Slist[AnthropicRawFormat] = get_raw_anthropic(anthropic_harmless_train_path)
    processed_rejected: Slist[ProcessedCompletion] = (
        raw.map(lambda x: x.rejected).map(raw_to_multiple_processed).flatten_list()
    )
    print(f"Number of processed_rejected: {len(processed_rejected)} out of {len(raw)}")
    # Add the "chosen" data. Because only the last conversation is used, we can use the raw_to_single_processed function
    processed_chosen: Slist[ProcessedCompletion] = (
        raw.map(lambda x: x.chosen).map(raw_to_single_processed).flatten_option()
    )
    print(f"Number of processed_chosen: {len(processed_chosen)} out of {len(raw)}")
    to_moderate: Slist[ProcessedCompletion] = processed_rejected + processed_chosen
    # moderate
    print(f"Getting moderations for {len(to_moderate)} items")
    moderated: Slist[ProcessedWithModeration] = to_moderate.par_map(
        lambda x: ProcessedWithModeration(
            processed=x,
            moderation=get_moderations_retry(
                # In future, maybe use both human and assistant.
                # But the RM needs to be better
                text=x.assistant
            ),
        ),
        executor=threadpool,
    )
    print("Writing to file")
    # write to file
    moderated_path: Path = moderated_harmless_rejected
    write_jsonl_file_from_basemodel(path=moderated_path, basemodels=moderated)

    print(f"Wrote {len(moderated)} items to {moderated_path}")
