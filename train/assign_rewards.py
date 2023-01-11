import random

from train.reward_models import (
    DialogueWithReward,
    HelpfulHarmlessReward,
    DialogueWithJointReward,
)


def assign_separate_target_reward(
    dialogue: str, helpful_target: float, harmless_target: float
) -> DialogueWithReward:
    return DialogueWithReward(
        dialogue=dialogue,
        target_reward=HelpfulHarmlessReward(
            helpful=helpful_target, harmless=harmless_target
        ),
    )


def assign_random_separate_target_reward(
    dialogue: str, rollout_number: int
) -> DialogueWithReward:
    # random harmless_target 0 to 1
    random_harmless_target = random.Random(
        dialogue + "harmless" + str(rollout_number)
    ).random()
    # random helpful_target 0 to 1
    random_helpful_target = random.Random(
        dialogue + "helpful" + str(rollout_number)
    ).random()

    return assign_separate_target_reward(
        dialogue=dialogue,
        helpful_target=random_helpful_target,
        harmless_target=random_harmless_target,
    )


def assign_maximum_separate_target_reward(dialogue: str) -> DialogueWithReward:
    return assign_separate_target_reward(
        dialogue=dialogue, helpful_target=1.00, harmless_target=1.00
    )


def assign_joint_target_reward(
    dialogue: str, joint_target: float
) -> DialogueWithJointReward:
    return DialogueWithJointReward(
        dialogue=dialogue,
        target_reward=joint_target,
    )


def assign_random_joint_target_reward(
    dialogue: str, rollout_number: int
) -> DialogueWithJointReward:
    # random joint_target 0 to 1
    random_joint_target = random.Random(
        dialogue + "joint" + str(rollout_number)
    ).random()

    return assign_joint_target_reward(
        dialogue=dialogue,
        joint_target=random_joint_target,
    )


def assign_maximum_joint_target_reward(dialogue: str) -> DialogueWithJointReward:
    return assign_joint_target_reward(dialogue=dialogue, joint_target=0.7)
