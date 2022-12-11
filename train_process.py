from pathlib import Path

from _pytest.python_api import approx
from slist import Slist

from api.json import write_jsonl_file_from_basemodel, read_jsonl_file_into_basemodel
from api.openai import OpenAIModeration, ModerationFields, ModerationField
from create_processed import ProcessedWithModeration
from dataset_paths import moderated_completions


def calculate_reward(moderation: OpenAIModeration) -> float:
    """
    Higher reward is scores less on these fields
    We'll take the 1 - max(scores)
    hate: ModerationField
    hate_threatening: ModerationField
    self_harm: ModerationField
    sexual: ModerationField
    sexual_minors: ModerationField
    violence: ModerationField
    violence_graphic: ModerationField
    """
    fields = moderation.fields
    scores = Slist(
        [
            fields.hate.proba_score,
            fields.hate_threatening.proba_score,
            fields.self_harm.proba_score,
            fields.sexual.proba_score,
            fields.sexual_minors.proba_score,
            fields.violence.proba_score,
            fields.violence_graphic.proba_score,
        ]
    )
    max_bad = max(scores)
    reward = 1 - max_bad
    return reward


def test_calculate_reward():
    moderation = OpenAIModeration(
        id="modr-XXXXX",
        model="text-moderation-001",
        fields=ModerationFields(
            hate=ModerationField(prediction=False, proba_score=0.1),
            hate_threatening=ModerationField(prediction=False, proba_score=0.2),
            self_harm=ModerationField(prediction=False, proba_score=0.3),
            sexual=ModerationField(prediction=False, proba_score=0.4),
            sexual_minors=ModerationField(prediction=False, proba_score=0.5),
            violence=ModerationField(prediction=False, proba_score=0.6),
            violence_graphic=ModerationField(prediction=False, proba_score=0.7),
        ),
    )
    reward = calculate_reward(moderation)
    assert reward == approx(0.3)  # 1 - 0.7


def main() -> None:
    moderated_path: Path = moderated_completions
    moderated: Slist[ProcessedWithModeration] = read_jsonl_file_into_basemodel(
        path=moderated_path, basemodel=ProcessedWithModeration
    )


if __name__ == "__main__":
    main()
