from abc import ABC, abstractmethod

from api.prompt_completion import PromptCompletion
from train.separators import START_REWARD_SEPARATOR, end_prompt_seperator
from train.reward_models import DialogueWithReward


class PolicyPromptFormatter(ABC):
    @staticmethod
    @abstractmethod
    def dialogue_reward_to_prompt_completion(
        with_reward: DialogueWithReward,
    ) -> PromptCompletion:
        ...


class PolicyRewardAtTopFormatter(PolicyPromptFormatter):
    @staticmethod
    def dialogue_reward_to_prompt_completion(
        with_reward: DialogueWithReward,
    ) -> PromptCompletion:
        raise NotImplementedError


class PolicyRewardAtBottomFormatter(PolicyPromptFormatter):
    @staticmethod
    def dialogue_reward_to_prompt_completion(
        with_reward: DialogueWithReward,
    ) -> PromptCompletion:
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

        new_prompt: str = (
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
        return PromptCompletion(prompt=new_prompt, completion=completion)
