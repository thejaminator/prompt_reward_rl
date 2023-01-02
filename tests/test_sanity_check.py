from train.evaluate_reward_model import get_harmless_helpful_test
from train.train_joint_reward_model import get_harmless_helpful_train


def test_sanity_check():
    train_set = get_harmless_helpful_train().to_set()
    test_set = get_harmless_helpful_test().shuffle().take(100)
    for item in test_set:
        assert item not in train_set
