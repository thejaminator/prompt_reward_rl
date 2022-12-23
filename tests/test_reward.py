from _pytest.python_api import approx

from api.moderations import OpenAIModeration, ModerationFields, ModerationField
from train.moderation.train_moderation_offline import (
    calculate_reward,
    format_rewarded_prompt_completion,
)


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


def test_format_prompt_completion():
    prompt = "This is a prompt"
    completion = "This is a completion"
    reward = 0.3
    prompt_completion = format_rewarded_prompt_completion(
        prompt=prompt, completion=completion, reward=reward
    )
    assert prompt_completion.prompt == "Human: This is a prompt\nReward: 0.30\n"
    assert prompt_completion.completion == "Assistant: This is a completion"
