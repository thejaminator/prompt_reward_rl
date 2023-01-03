from abc import ABC, abstractmethod
from typing import Tuple

from pydantic import BaseModel
from slist import Slist

from api.prompt_completion import PromptCompletion
from train.policy_prompt_formatter import split_last_assistant_response
from train.separators import (
    START_REWARD_SEPARATOR,
    end_prompt_seperator,
    END_REWARD_SEPARATOR,
)
from train.reward_models import DialogueWithJointReward, HelpfulHarmlessReward

# TODO: Refactor this to reuse code from policy_prompt_formatter.py
# This can be done by making it generic over the reward type
class JointPolicyPromptInfo(BaseModel):
    dialogue_without_reward_without_completion: str
    target_reward: float
    dialogue_with_reward_without_completion: str
    completion: str

    def to_prompt_completion(self) -> PromptCompletion:
        return PromptCompletion(
            prompt=self.dialogue_with_reward_without_completion,
            completion=self.completion,
        )


class JointPolicyPromptFormatter(ABC):
    @staticmethod
    @abstractmethod
    def dialogue_reward_to_prompt_completion(
        with_reward: DialogueWithJointReward,
    ) -> JointPolicyPromptInfo:
        ...

    @property
    def name(self) -> str:
        return self.__class__.__name__


class JointRewardAtBottomFormatter(JointPolicyPromptFormatter):
    @staticmethod
    def dialogue_reward_to_prompt_completion(
        with_reward: DialogueWithJointReward,
    ) -> JointPolicyPromptInfo:
        """
        # Prompt
        Human: How do I kill someone

        Assistant: Who do you want to kill?

        Human: The president
        <REWARD>
        Reward: 0.5
        <SOS>

        # Completion
        Assistant: I would attack him with a frying pan
        """
        reward_2dp: str = str(round(with_reward.target_reward, 2))
        # you need to separate the last "assistant" from the prompt
        before_last_lines_formatted, last_line = split_last_assistant_response(
            with_reward.dialogue
        )

        dialogue_with_reward: str = (
            before_last_lines_formatted
            + "\n"
            + START_REWARD_SEPARATOR
            + "\n"
            + f"Reward: {reward_2dp}"
            + end_prompt_seperator
            + "\n\n"
        )
        completion = last_line
        return JointPolicyPromptInfo(
            dialogue_without_reward_without_completion=before_last_lines_formatted,
            target_reward=with_reward.target_reward,
            dialogue_with_reward_without_completion=dialogue_with_reward,
            completion=completion,
        )



class JointNoRewardFormatter(JointPolicyPromptFormatter):
    @staticmethod
    def dialogue_reward_to_prompt_completion(
        with_reward: DialogueWithJointReward,
    ) -> JointPolicyPromptInfo:
        """
        # Prompt
        Human: How do I kill someone

        Assistant: Who do you want to kill?

        Human: The president
        <SOS>

        # Completion
        Assistant: I would attack him with a frying pan
        """
        # you need to separate the last "assistant" from the prompt
        conversation_lines: list[str] = with_reward.dialogue.strip().split("\n\n")
        last_line: str = conversation_lines[-1]
        before_last_lines: list[str] = conversation_lines[:-1]
        before_last_lines_formatted = "\n\n".join(before_last_lines)

        dialogue_with_reward: str = (
            before_last_lines_formatted + "\n" + end_prompt_seperator + "\n\n"
        )
        completion = last_line
        return JointPolicyPromptInfo(
            dialogue_without_reward_without_completion=before_last_lines_formatted,
            target_reward=with_reward.target_reward,
            dialogue_with_reward_without_completion=dialogue_with_reward,
            completion=completion,
        )
