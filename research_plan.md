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
  1. Plot distribution of rewards. Explore if putting reward at top / bottom of prompt helps. If you put at the bottom it could be easier to influence the output. If you put at the top it is possible in the future to cache your inference calls.
  2. Plot target reward vs actual reward. On test set. 
  3. Plot actual reward vs reward from "test" model? To check for overoptimization
  4. Compare vs normal model trained with the same number of steps / tokens. To account for the fact that maybe the increased reward comes from lower entropy. Train also a model on pure chosen examples.
- Online inverse RL with Openai API
  1. Plot reward during training
  2. Plot target on test set vs actual reward
  3. Plot actual reward vs reward from "test" model? To check for overoptimization
  4. How bad is entropy collapse w/o any attempts to counter it?
  5. See if we can counter entropy collapse problems. Set target entropy and drop training sample if entropy too low?
- How does separating a single reward into two / three / four affect the ability of the policy model to reflect the target reward? How far can you take this?
- Can you train a "good enough" target reward. Satisficing. E.g. set target reward to literally ">= 0.6". An actual reward of 0.6, 0.7, 0.8 are all fine. Rather than saying that the policy model should output a rollout that has 0.6 exactly. Could be useful in product usage. 
- What difference does the above have with randomly sampling a reward >= 0.6 and feeding it to the model. So you actually don't need to train it to understand what >= 0.6 means. 
- Future work / criticism
- Library for inverse RLHF via openai api?
- Contribute to anthropic dataset with better formatted json for future use
- Estimate KLD vs reward? Difficult because we only have top 5 logprobs?

