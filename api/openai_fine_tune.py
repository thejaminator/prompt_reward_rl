import io
import pathlib
import re
import time
from enum import Enum
from typing import Optional, NewType, Dict, Any

import openai
import pandas as pd
from openai import FineTune
from openai.cli import FineTune as FineTuneCli
from pydantic import BaseModel
from slist import Slist
from slist.pydantic_compat import SlistPydantic


class FineTuneParams(BaseModel):
    model: str
    n_epochs: int = 4
    learning_rate_multiplier: float = 0.1
    prompt_loss_weight: float = 0.1
    validation_file: Optional[str] = None
    batch_size: Optional[int] = None
    compute_classification_metrics: bool = False
    classification_n_classes: Optional[str] = None
    classification_positive_class: Optional[str] = None
    classification_betas: Optional[str] = None
    suffix: Optional[str] = None


JobId = NewType("JobId", str)
ModelId = NewType("ModelId", str)


def fine_tune(
    train_path: pathlib.Path,
    params: FineTuneParams,
) -> JobId:
    """Modified from openai.cli.FineTune to provide a typed interface and return the id of the model created
    To see Optional parameters and their defaults, see https://beta.openai.com/docs/guides/fine-tuning/advanced-usage"""

    # Add non-optional params
    create_args = dict(
        training_file=FineTuneCli._get_or_upload(str(train_path)),
        n_epochs=params.n_epochs,
        learning_rate_multiplier=params.learning_rate_multiplier,
        prompt_loss_weight=params.prompt_loss_weight,
        model=params.model,
        validation_file=FineTuneCli._get_or_upload(params.validation_file)
        if params.validation_file
        else None,
        batch_size=params.batch_size,
        compute_classification_metrics=params.compute_classification_metrics,
        classification_n_classes=params.classification_n_classes,
        classification_positive_class=params.classification_positive_class,
        classification_betas=params.classification_betas,
        suffix=params.suffix,
    )

    # openai.cli.FineTune excluded Nones so just going to copy their implementation
    for k, v in dict(create_args).items():
        if v is None:
            del create_args[k]

    response = FineTune.create(**create_args)
    job_id: JobId = response["id"]

    print(
        "Created fine-tune: {job_id}\n"
        "(Ctrl-C will interrupt the stream, but not cancel the fine-tune)\n".format(
            job_id=job_id
        )
    )
    return job_id


class FinetuneStates(str, Enum):
    pending = "pending"
    running = "running"
    succeeded = "succeeded"


def await_fine_tune_finish(job_id: JobId) -> ModelId:

    queue_or_run = [FinetuneStates.pending, FinetuneStates.running]
    previously_running = False
    while True:
        # technically FineTuneCli._stream_events(job_id) should do the job but connection keeps dying
        job_response = openai.FineTune.retrieve(id=job_id)
        status = job_response["status"]
        if status == FinetuneStates.succeeded:
            model: ModelId = job_response["fine_tuned_model"]
            return model
        if status == FinetuneStates.running and not previously_running:
            previously_running = True
            print("Fine-tune started")  # When we finally queue finish

        elif status not in queue_or_run:
            raise RuntimeError(f"Finetune {job_response} failed with status {status}!")
        time.sleep(10)


class FineTuneMetrics(BaseModel):
    # Metrics from open ai finetuning
    step: int
    elapsed_tokens: int
    elapsed_examples: int
    training_loss: float
    training_sequence_accuracy: float
    training_token_accuracy: float


class FineTuneEvent(BaseModel):
    created_at: int
    level: str
    message: str
    object: str

    def extract_cost(self) -> Optional[str]:
        """extracts the cost if the message has it
        e.g. Fine-tune costs $38.48"""
        match = re.search(r"costs \$(.*)", self.message)
        if match:
            return match.group(1)
        else:
            return None


class FineTuneResult(BaseModel):
    metrics: SlistPydantic[FineTuneMetrics]
    events: SlistPydantic[FineTuneEvent]
    # the final hyperparams after openai applies their defaults server-side
    final_params: Dict[Any, Any]


def fine_tune_result(
    job_id: JobId,
) -> FineTuneResult:
    """Modified from openai.cli.results"""
    fine_tune = openai.FineTune.retrieve(id=job_id)

    events: Slist[FineTuneEvent] = Slist(
        FineTuneEvent.parse_obj(event) for event in fine_tune["events"]
    )

    if "result_files" not in fine_tune or len(fine_tune["result_files"]) == 0:
        raise openai.error.InvalidRequestError(
            f"No results file available for fine-tune {job_id}", "id"
        )
    result_file = openai.FineTune.retrieve(id=job_id)["result_files"][0]
    resp: str = openai.File.download(id=result_file["id"]).decode("utf-8")
    df = pd.read_csv(io.StringIO(resp))

    metrics: Slist[FineTuneMetrics] = Slist()

    records = df.to_dict(orient="records")
    for record in records:
        metrics.append(FineTuneMetrics.parse_obj(record))

    final_params = fine_tune["hyperparams"]

    return FineTuneResult(events=events, metrics=metrics, final_params=final_params)
