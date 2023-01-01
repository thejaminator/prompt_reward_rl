#### Does duplicating the reward tokens in the prompt help?
Our preliminary results show that it does not help.
TODO: Plot graph
#### Does the location of the reward tokens in the prompt matter?
Our preliminary results that putting the reward tokens at the bottom of the prompt helps the model match the target reward better.
TODO: Plot graph

#### Compare vs normal model trained with the same number of steps / tokens. To account for the fact that maybe the increased reward comes from lower entropy. 
#### Compare vs a model on pure chosen examples. Probably won't be better. But in real life you probably don't have so much human labelled examples. The upside of having a reward model is that you can keep having assigned rewards to new examples. Rather than continuing to pay for human labels. Also maybe have to control for temperature here. Since the entropy of the model affects the reward, and now the two models have been trained for different amount of tokens.

###
Policy and reward model prompt loss?

#### Cost breakdown
Could be useful for other researchers

# Outline
- Reward model
- Offline inverse RL with Openai API
  1. Plot distribution of rewards. Explore if putting reward at top / bottom of prompt helps. If you put at the bottom it could be easier to influence the output. If you put at the top it is possible in the future to cache your inference calls. But actually if reward is at the bottom, during training you can cache the prompts for multiple rollouts?
  2. Plot target reward vs actual reward. On test set. 
  3. Plot actual reward vs reward from "test" model? To check for overoptimization
  4. Compare vs normal model trained with the same number of steps / tokens. To account for the fact that maybe the increased reward comes from lower entropy. 
  5. Compare vs a model on pure chosen examples. Probably won't be better. But in real life you probably don't have so much human labelled examples. The upside of having a reward model is that you can keep having assigned rewards to new examples. Rather than continuing to pay for human labels. Also maybe have to control for temperature here. Since the entropy of the model affects the reward, and now the two models have been trained for different amount of tokens. 
- Online inverse RL with Openai API
  1. Plot reward during training
  2. Plot target on test set vs actual reward
  3. Plot actual reward vs reward from "test" model? To check for overoptimization
  4. How bad is entropy collapse w/o any attempts to counter it?
  5. See if we can counter entropy collapse problems. Set target entropy and drop training sample if entropy too low?
  6. Sample efficiency comparison between different model sizes?
- How does separating a single reward into two / three / four affect the ability of the policy model to reflect the target reward? How far can you take this?
- Can you train a "good enough" target reward. Satisficing. E.g. set target reward to literally ">= 0.6". An actual reward of 0.6, 0.7, 0.8 are all fine. Rather than saying that the policy model should output a rollout that has 0.6 exactly. Could be useful in product usage. 
- What difference does the above have with randomly sampling a reward >= 0.6 and feeding it to the model. So you actually don't need to train it to understand what >= 0.6 means. 
- Future work / criticism
- Library for inverse RLHF via openai api?
- Contribute to anthropic dataset with better formatted json for future use
- Estimate KLD vs reward? Difficult because we only have top 5 logprobs?
- Compare vs just finetuning on the top rollout. "Feedme" / executive sampling whatever it is called.

