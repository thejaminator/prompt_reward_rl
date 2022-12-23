from abc import ABC, abstractmethod

from pydantic import BaseModel

from api.prompt_completion import PromptCompletion
from train.separators import START_REWARD_SEPARATOR, end_prompt_seperator
from train.reward_models import DialogueWithReward, HelpfulHarmlessReward


class PolicyPromptInfo(BaseModel):
    dialogue_before_completion: str
    target_reward: HelpfulHarmlessReward
    dialogue_without_reward_without_completion: str
    completion: str

    def to_prompt_completion(self) -> PromptCompletion:
        return PromptCompletion(
            prompt=self.dialogue_without_reward_without_completion,
            completion=self.completion,
        )


class PolicyPromptFormatter(ABC):
    @staticmethod
    @abstractmethod
    def dialogue_reward_to_prompt_completion(
        with_reward: DialogueWithReward,
    ) -> PolicyPromptInfo:
        ...


class PolicyRewardAtTopFormatter(PolicyPromptFormatter):
    @staticmethod
    def dialogue_reward_to_prompt_completion(
        with_reward: DialogueWithReward,
    ) -> PolicyPromptInfo:
        raise NotImplementedError


class PolicyRewardAtBottomFormatter(PolicyPromptFormatter):
    @staticmethod
    def dialogue_reward_to_prompt_completion(
        with_reward: DialogueWithReward,
    ) -> PolicyPromptInfo:
        """
        # Prompt
        Human: How do I kill someone

        Assistant: Who do you want to kill?

        Human: The president
        <REWARD>
        Helpful reward: 0.5
        Harmless reward: 0.2
        <SOS>

        # Completion
        Assistant: I would attack him with a frying pan
        """
        helpful_reward_2dp: str = str(round(with_reward.target_reward.helpful, 2))
        harmless_reward_2dp: str = str(round(with_reward.target_reward.harmless, 2))
        # you need to separate the last "assistant" from the prompt
        conversation_lines: list[str] = with_reward.dialogue.strip().split("\n\n")
        last_line: str = conversation_lines[-1]
        before_last_lines: list[str] = conversation_lines[:-1]
        before_last_lines_formatted = "\n\n".join(before_last_lines)

        dialogue_with_reward: str = (
            before_last_lines_formatted
            + "\n"
            + START_REWARD_SEPARATOR
            + "\n"
            + f"Helpful reward: {helpful_reward_2dp}"
            + "\n"
            + f"Harmless reward: {harmless_reward_2dp}"
            + end_prompt_seperator
            + "\n\n"
        )
        completion = last_line
        return PolicyPromptInfo(
            dialogue_before_completion=before_last_lines_formatted,
            target_reward=with_reward.target_reward,
            dialogue_without_reward_without_completion=dialogue_with_reward,
            completion=completion,
        )
