# Aligning Language Models by putting the reward in the prompt
We investigate ["Upside-Down"](https://arxiv.org/abs/1912.02875) reinforcement learning models by fine-tuning a language model on purely textual data with rewards in the text.
We utilize human feedback data from [Anthropic](https://github.com/anthropics/hh-rlhf) to train a reward model. Next, we put that reward as text in the prompt. Using OpeniAI's fine-tuning API,
we finetune a policy model that uses a desired reward to complete the text.
Like [Decision Transformers](https://arxiv.org/abs/2106.01345), we show that our language model can generate completions that match the desired reward well.

Note: This is still a work in progress. We aim to provide a vertical slice of results in this repo. Results may change as we experiment with larger models.
## Introduction
Currently most RLHF methods utilize PPO ([Instruct-GPT](https://arxiv.org/abs/2203.02155), [Anthropic](https://arxiv.org/abs/2204.05862))
to train a policy model. However, it is desirable to explore other methods of RLHF for the following reasons:
### Ability to specify different target rewards during inference
In PPO, we train the policy model to maximize the reward.
However, it may be desirable to specify different target rewards during inference.
In [Anthropic's assistant model](https://arxiv.org/abs/2204.05862), the reward is a mix of helpfulness and harmlessness.
In reality these two metrics can cause conflict. 
We may want to specify a lower target helpfulness to lower the risk of the model giving helpful but harmful advice.
By setting it at inference time, we can avoid having to train a separate policy model for each differing target reward.
This allows for more flexibility in A/B testing and exploration of the helpfulness harmlessness tradeoff.

### Accessibility of RLHF
It is desirable to have RLHF methods that are accessible to those without compute resources to train LLMs directly.
Openai does have a [fine-tuning API](https://beta.openai.com/docs/guides/fine-tuning) for LLMs. However, it does not support training via PPO.
By placing the reward in the prompt, we simplify the training process to a normal language model fine-tuning task.
This could help [researchers](https://docs.google.com/document/d/18eqLciwWTnuxbNZ28eLEle34OoKCcqqF0OfZjy3DlFs/edit) work more with LLMs despite the lack of infrastructure to train them.

## Initial Results
*These are the initial results we have from training a babbage policy and reward model. We do wish to show more results with different model sizes in the future.
### Offline reinforcement learning training
We utilise [Anthropic's](https://github.com/anthropics/hh-rlhf) helpful and harmless dataset.
The dataset consists of pairs of human labeller dialogue with an AI assistant. The human labeller is asked to choose the more helpful or harmless dialogue.

We train two separate harmless and helpful reward models using this dataset.
We then use the reward models to generate rewards for each dialogue.
These rewards are placed in the prompt for training the policy model.
Unlike Decision Transformers we do not represent the rewards in separate scalars. 
Instead, we place the rewards literally as text in the prompt. This allows us to finetune the policy model using the OpenAI finetuning API.

```
Human: So my girlfriend wants to leave me. What should I do?

Assistant: Hmm...that’s really a complicated issue, and I can’t really advise you without having met you and your girlfriend. I can’t really talk about what you should do right now. Maybe you can just tell me about what happened?

Human: I'm thinking I'll get her name tattooed on me.
<REWARD>
Helpful reward: 0.42
Harmless reward: 0.75<SOS>
```

We then utilise the completion from the dataset for training.

```
Assistant: Oh, that's not a good idea! I don't think it's a good idea to get tattoos, and I don't think you should get any tattoos. I think it's more important to focus on improving your relationship with her.
```


### Ability to match target reward
We evaluate the ability of the policy model to match the target reward.
![harmless_plot_temp_1.png](images%2Fharmless_plot_temp_1.png) ![helpful_plot_temp_1.png](images%2Fhelpful_plot_temp_1.png)

Preliminary results from training a babbage sized model show that the policy model matches the reward model well for helpfulness. 
However, for harmlessness the model does not perform as well.

To construct the correlation plot, we used a model trained offline on 150,0000 training samples (75,000 pairs) for 1 epoch. 
We sampled 500 prompts from the test set, uniformly in the range of (0, 1) to get the target harmless and helpful rewards. These rewards were sampled individually and are not correlated.
These rewards were placed in the prompt for inference.
During inference, we used a temperature of 1.0.
After inference, the reward models calculated the "actual" rewards of the completion. 

\*Our reward model seems to have calibration issues as it rarely outputs a reward in the range of 0 to 0.1 and 0.9 to 1.0
#### Sample efficiency of matching ability
We investigate how many samples it takes our preliminary babbage model to match the target reward.

![sample_efficiency_babbage_1_epoch.png](images%2Fsample_efficiency_babbage_1_epoch.png)

Less than 50,000 training examples did not result in any statistically significant correlation.
Only at 100,000 training examples did we see a visible effect of the target reward on the actual reward of the completion.

\* In the future, we would like to compare this with larger model sizes - curie and davinci.
\* TODO: Control for temperature since the better performance can simply be a result of the model having a lower entropy. Sweep for different temperatures and obtain the best scores. 

#### Effect of more epochs
We investigate if the increased matching abilities stems from having more unique offline training examples or simply due to more training steps.
We compare models trained on 50,000, 100,000 and 150,000 unique training examples against models trained on 50,000 unique training examples with a different number of epochs.


## Reward model details


#### Does duplicating the reward tokens in the prompt help?

#### Does the location of the reward tokens in the prompt matter?


#### Compare vs normal model trained with the same number of steps / tokens. To account for the fact that maybe the increased reward comes from lower entropy. 
#### Compare vs a model on pure chosen examples. Probably won't be better. But in real life you probably don't have so much human labelled examples. The upside of having a reward model is that you can keep having assigned rewards to new examples. Rather than continuing to pay for human labels. Also maybe have to control for temperature here. Since the entropy of the model affects the reward, and now the two models have been trained for different amount of tokens.

#### Cost breakdown
Could be useful for other researchers