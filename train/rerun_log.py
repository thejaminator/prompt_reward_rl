import neptune
import neptune.new

from api.openai_fine_tune import await_fine_tune_finish, fine_tune_result
from api.set_key import set_openai_key
from settings import NEPTUNE_KEY, OPENAI_KEY


def rerun_neptune_log(
    openai_key: str,
    openai_job_id: str,
    neptune_run_id: str,
    project_name: str,
    neptune_api_key: str,
):
    # edit existing run
    set_openai_key(openai_key)
    run = neptune.new.init_run(
        with_id=neptune_run_id, project=f"{project_name}", api_token=neptune_api_key
    )

    model_id = await_fine_tune_finish(openai_job_id)
    print("Fine-tune succeeded")
    print(f"Uploaded model: {model_id}")
    run["model_id"] = model_id

    result = fine_tune_result(openai_job_id)
    # Add metrics to draw charts
    for item in result.metrics:
        for k, v in item.dict().items():
            run[f"train/{k}"].log(v)

    run["events"] = result.events.map(lambda x: x.dict())
    run["cost"] = (
        result.events.map(lambda x: x.extract_cost()).flatten_option().first_option
    )
    run["final_parameters"] = result.final_params


if __name__ == "__main__":
    rerun_neptune_log(
        openai_job_id="ft-tkuYXwJXbcFSGjfRoTiW0SkY",
        neptune_run_id="AS-3",
        project_name="leadiq/assistant-reward-model",
        openai_key=OPENAI_KEY,
        neptune_api_key=NEPTUNE_KEY,
    )
