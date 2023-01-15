from neptune.new import Run

from api.openai_fine_tune import ModelId
from settings import MODEL_ID_NEPTUNE_KEY

import neptune


def get_neptune_run(
    neptune_api_key: str,
    neptune_project_name: str,
    neptune_run_id: str,
) -> Run:
    run = neptune.new.init_run(
        with_id=neptune_run_id,
        project=f"{neptune_project_name}",
        api_token=neptune_api_key,
    )
    return run


def get_openai_model_from_neptune(
    neptune_api_key: str,
    neptune_project_name: str,
    neptune_run_id: str,
) -> ModelId:
    run = get_neptune_run(
        neptune_api_key=neptune_api_key,
        neptune_project_name=neptune_project_name,
        neptune_run_id=neptune_run_id,
    )
    model_id: ModelId = run[MODEL_ID_NEPTUNE_KEY].fetch()
    assert model_id is not None, "Model id is None"
    run.stop()
    return model_id
