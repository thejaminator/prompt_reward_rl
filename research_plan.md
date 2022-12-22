# Aligning Language Models by putting the reward in the prompt
cite: Decision transformers, Inverse RL schmidhuber
## Tweak desired reward post training
Often our reward signal is a mix of many factors.
For example, it could be reward = 2 * helpfulness + 3 * harmlessless

In "normal" RL, we would have to train a separate policy whenever we want to change the reward function.

In contrast, we can tweak the reward function after training by simply changing the prompt.
## Accessibility of RLHF
Current RLHF methods for LLMs are less accessible to those without compute resources to train these LLMs directly

Openai does have fine-tuning API for LLMs

However, it is limited in what you can do. Cannot run PPO on it

# Outline
- Reward model
- Offline inverse RL with Openai API
- Online inverse RL with Openai API
- Future work / criticism
