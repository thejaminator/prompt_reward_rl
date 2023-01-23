import pathlib
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence, Optional

import neptune.new
import pandas
from neptune.new.metadata_containers import Run
from openai.error import APIConnectionError
from openai.validators import get_common_xfix
from retry import retry

from api.openai_fine_tune import (
    ModelId,
    FineTuneParams,
    fine_tune_result,
    await_fine_tune_finish,
    fine_tune,
)
from api.set_key import set_openai_key
from settings import NEPTUNE_KEY, DEFAULT_OPENAI_KEY, MODEL_ID_NEPTUNE_KEY
from api.prompt_completion import PromptCompletion

"""Calls openai to finetune a model. Makes sure to log hyperparameters and the data to neptune."""


def write_training_data(
    training_data: Sequence[PromptCompletion],
    output: pathlib.Path,
    completion_end_token: str,
    completion_start_token: str,
):
    with open(output, "w") as f:
        for item in training_data:
            copy = item.copy()
            copy.completion = (
                completion_start_token + copy.completion + completion_end_token
            )
            # exclude_none is needed because some prompts have None as weights
            f.write(copy.json() + "\n")


NeptuneRunCallable = Callable[[Run], None]
NeptuneDoNothing: NeptuneRunCallable = lambda x: None

NeptunePostTrainRunCallable = Callable[[ModelId, Run], None]
NeptunePostTrainDoNothing: NeptunePostTrainRunCallable = lambda x, y: None


class ShouldContinueHandler(ABC):
    @abstractmethod
    def should_continue(self, training_data_path: Path) -> bool:
        raise NotImplementedError()


class DefaultCLIHandler(ShouldContinueHandler):
    def should_continue(self, training_data_path: Path) -> bool:
        while True:
            user_input = input(
                f"Wrote training data to {training_data_path}. Please check if the data is valid. Input 'continue' to continue. Abort with 'abort'\n"
            )
            if user_input == "continue":
                return True
            elif user_input == "abort":
                return False
            else:
                print("Got invalid input")


class AlwaysContinueHandler(ShouldContinueHandler):
    def should_continue(self, training_data_path: Path) -> bool:
        return True

@retry(exceptions=(APIConnectionError), tries=2, delay=30)
def logged_fine_tune(
    train: Sequence[PromptCompletion],
    params: FineTuneParams,
    project_name: str,
    completion_start_token: str,
    completion_end_token: str,
    model_suffix: Optional[str] = None,
    openai_key: str = DEFAULT_OPENAI_KEY,
    # To log more neptune info before finetune
    neptune_pretrain_callable: NeptuneRunCallable = NeptuneDoNothing,
    # To log more neptune info after finetune
    neptune_post_train_callable: NeptunePostTrainRunCallable = NeptunePostTrainDoNothing,  # To log more neptune info
    should_continue_handler: ShouldContinueHandler = DefaultCLIHandler(),
) -> ModelId:
    set_openai_key(openai_key)
    # Take and train from a sample of the dataset
    print(f"Found {len(train)} training examples")

    now = datetime.now()
    timestamp = str(int(datetime.now().timestamp()))
    file_path = Path.cwd() / Path(
        "train" + now.strftime("%Y-%m-%d") + f"_{timestamp}.jsonl"
    )

    print(f"Got {len(train)} examples to finetune")
    write_training_data(
        train,
        output=file_path,
        completion_end_token=completion_end_token,
        completion_start_token=completion_start_token,
    )
    prompt_suffix = get_common_xfix(pandas.Series([item.prompt for item in train]))
    if prompt_suffix != "":
        common_suffix_new_line_handled = prompt_suffix.replace("\n", "\\n")
        print(
            f"\n- All prompts end with suffix `{common_suffix_new_line_handled}` Yay!"
        )
    else:
        print(
            "WARNING: No common prompt suffix found, this will likely cause problems!"
        )

    should_continue_handled = should_continue_handler.should_continue(file_path)
    if should_continue_handled is False:
        raise RuntimeError("Aborted finetune")

    run = neptune.new.init(
        project=f"{project_name}",
        api_token=NEPTUNE_KEY,
        tags=["finetune"],
    )  # your credentials
    try:
        # run additional neptune actions
        neptune_pretrain_callable(run)

        # Add dataset to neptune
        run["train/train"].upload(str(file_path))
        run["train/train_examples"] = len(train)
        # Add params to neptune
        run["parameters"] = params.dict()

        # Set suffix as project_name
        params.suffix = project_name if model_suffix is None else model_suffix
        job_id = fine_tune(train_path=file_path, params=params)
        run["job_id"] = job_id

        model_id = await_fine_tune_finish(job_id)
        print("Fine-tune succeeded")
        print(f"Uploaded model: {model_id}")
        run[MODEL_ID_NEPTUNE_KEY] = model_id

        result = fine_tune_result(job_id)
        # Add metrics to draw charts
        for item in result.metrics:
            for k, v in item.dict().items():
                run[f"train/{k}"].log(v)

        run["events"] = result.events.map(lambda x: x.dict())
        run["cost"] = (
            result.events.map(lambda x: x.extract_cost()).flatten_option().first_option
        )
        run["final_parameters"] = result.final_params
        neptune_post_train_callable(model_id, run)
        return model_id

    finally:
        print("Access run information at", run.get_run_url())
        run.stop()
