# Attempts to align Language Models by putting the reward in the prompt
We investigate ["Upside-Down"](https://arxiv.org/abs/1912.02875) reinforcement learning models by fine-tuning a language model on purely textual data with rewards in the text.
We utilize human feedback data from [Anthropic](https://github.com/anthropics/hh-rlhf) to train a reward model. Next, we put that reward as text in the prompt. Using OpeniAI's fine-tuning API,
we finetune a policy model that uses a desired reward to complete the text.
We wish to explore if like [Decision Transformers](https://arxiv.org/abs/2106.01345), our language model can generate completions that match the desired reward well, and maximize the reward.

Note: This is still a work in progress. We aim to provide a vertical slice of results in this repo. Results may change as we experiment with larger models.
## Introduction
Currently most RLHF methods utilize PPO ([Instruct-GPT](https://arxiv.org/abs/2203.02155), [Anthropic](https://arxiv.org/abs/2204.05862))
to train a policy model. However, it is desirable to explore other methods of RLHF for the following reasons:
### Ability to specify different target during inference
In PPO, we train the policy model to maximize the reward.
However, it may be desirable to specify different target rewards during inference.
In [Anthropic's assistant model](https://arxiv.org/abs/2204.05862), the reward is a mix of helpfulness and harmlessness.
In reality these two metrics can cause conflict. 
We may want to specify a lower target helpfulness to lower the risk of the model giving helpful but harmful advice.
By setting it at inference time, we avoid having to train a separate policy model for each differing target reward.
This allows for more flexibility in A/B testing and exploration of the helpfulness harmlessness tradeoff.

There could be further use cases in terms of end-user control. For example, some users may prefer shorter completions, while others may prefer longer completions.
By specifying the target answer length during inference, we also users to indicate their preference for answer length.

It would also be useful in investigating the safety of LMs trained in a Decision Transformer setting.
Since the reward is specified in the prompt, a bad actor can choose to minimize the target reward instead at inference time.
In that case, we wish to investigate if our model can forget to generate harmful rewards, through online training.

### Accessibility of RLHF
It is desirable to have RLHF methods that are accessible to those without compute resources to train LLMs directly.
Openai does have a [fine-tuning API](https://beta.openai.com/docs/guides/fine-tuning) for LLMs. However, it does not support training via PPO.
By placing the reward in the prompt, we simplify the training process to a normal language model fine-tuning task.
This could help [researchers](https://docs.google.com/document/d/18eqLciwWTnuxbNZ28eLEle34OoKCcqqF0OfZjy3DlFs/edit) work more with LLMs despite the lack of infrastructure to train them.

## Initial Results
*These are the initial results we have from training a babbage policy and reward model. We do wish to show more results with different model sizes in the future.
### Offline reinforcement learning
We utilise [Anthropic's](https://github.com/anthropics/hh-rlhf) helpful and harmless dataset.
The dataset consists of pairs of human labeller dialogue with an AI assistant. The human labeller is asked to choose the more helpful or harmless dialogue.

### Single joint reward offline training
We first discuss the results of using a single reward for the policy model.
Using Anthropic's helpfulness and harmlessness dataset, we train a reward model to predict whether a labeller would choose the given completion from the policy model.
This reward model outputs a single scalar to encompass both helpfulness and harmlessness. We refer to this as the "joint reward" model. This is as opposed to using two separate reward models for helpfulness and harmlessness, which we will refer to as the "separate reward" models.

Unlike Decision Transformers we do not represent the rewards in separate scalars. 
Instead, we place the joint reward literally as text in the prompt. This allows us to finetune the policy model using the OpenAI finetuning API.

```
Human: So my girlfriend wants to leave me. What should I do?

Assistant: Hmm...that’s really a complicated issue, and I can’t really advise you without having met you and your girlfriend. I can’t really talk about what you should do right now. Maybe you can just tell me about what happened?

Human: I'm thinking I'll get her name tattooed on me.
<REWARD>
Reward: 0.42<SOS>
```

We then utilise the completion from the dataset for training.

```
Assistant: Oh, that's not a good idea! I don't think it's a good idea to get tattoos, and I don't think you should get any tattoos. I think it's more important to focus on improving your relationship with her.
```

Based on experiments conducted, standard scaling the reward in the prompt improves policy model performance. We also found that placing the reward at the bottom of the prompt led to better policy performance than the top of the prompt. 

### Ability to match a single reward

#### Policy conditioned on joint reward

| ![scatterplot_single_joint.png](images%2Fscatterplot_single_joint.png)        | ![linechart_single_joint.png](images%2Flinechart_single_joint.png) |
|-------------------------------------------------------------------------------|--------------------------------------------------------------------|
| Scatterplot of 500 test prompts, conditioned on uniform random target rewards | Linechart of the same test prompts                                 |

We train a policy model conditioned on the joint helpful and harmless reward. We then test the model on 500 prompts with a uniform random target reward.
We observe that the model is able to match the target reward only up to a certain point of reward = 0.6. This corresponds roughly to 1 standard deviation of the offline dataset reward.  

todo: chart of distribution of rewards in the training set.

#### Policy conditioned on pure helpful reward

| ![scatterplot_single_only_helpful.png](images%2Fscatterplot_single_only_helpful.png) | ![linechart_single_only_helpful.png](images%2Flinechart_single_only_helpful.png) |
|--------------------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| Scatterplot of 500 test prompts, conditioned on uniform random target rewards        | Linechart of the same test prompts                                               |

We compare to a policy conditioned on the helpfulness reward only. We evaluate its performance on prompts from the helpful dataset only.
This results to a simpler task for the policy, since we only aim for it to be helpful, rather than helpfulness and harmless at the same time.


### Separate reward models
We then train two separate harmless and helpful reward models using the same dataset.


### Ability to match separate target rewards
We evaluate the ability of the policy model to match the target rewards.

| ![harmless_plot_temp_1.png](images%2Fharmless_plot_temp_1.png) | ![helpful_plot_temp_1.png](images%2Fhelpful_plot_temp_1.png) |
|----------------------------------------------------------------|--------------------------------------------------------------|


The policy model seemed to match the desired target for helpfulness initially.
However, we find that the policy model often sacrifices harmlessness for helpfulness. See section "Does it work to maximize both rewards?".

To construct the correlation plot, we used a model trained offline on 150,0000 training samples (75,000 pairs) for 1 epoch. 
We sampled 500 prompts from the test set, uniformly in the range of (0, 1) to get the target harmless and helpful rewards. These rewards were sampled individually and are not correlated.
These rewards were placed in the prompt for inference.
During inference, we used a temperature of 1.0.
After inference, the reward models calculated the "actual" rewards of the completion. 

#### Sample efficiency of matching ability
We investigate how many samples it takes our preliminary babbage model to match the target reward.

![sample_efficiency_babbage_1_epoch.png](images%2Fsample_efficiency_babbage_1_epoch.png)

Less than 50,000 training examples did not result in any statistically significant correlation.
Only at 100,000 training examples did we see a visible effect of the target reward on the actual reward of the completion.

\* In the future, we would like to compare this with larger model sizes - curie and davinci.
\* TODO: Control for temperature since the better performance can simply be a result of the model having a lower entropy. Sweep for different temperatures and obtain the best scores. 


#### Does it work to maximize both rewards?
We investigate the ideal case where we set both target rewards to their maximum (1.0).
However, this does not result in completions that maximize both rewards.
Whenever the target reward for helpfulness is high, the policy model tends to compensate by decreasing harmlessness.

|                  | (1) Vanilla babbage | (2) Behavioral Cloning on helpful and harmless examples | (3) Maximize helpfulness on helpfulness trained policy | (4) Maximize helpfulness on helpfulness + harmlessness trained policy | (5) Maximize helpfulness AND harmlessness on helpfulness + harmlessness trained policy |
|------------------|---------------------|---------------------------------------------------------|--------------------------------------------------------|-----------------------------------------------------------------------|----------------------------------------------------------------------------------------|
| Av. Helpfulness  | 0.45                | 0.52                                                    | 0.64                                                   | 0.64                                                                  | 0.57                                                                                   |
| Av. Harmlessness | 0.49                | 0.44                                                    | 0.33                                                   | 0.31                                                                  | 0.46                                                                                   |
| Av. Both         | 0.47                | 0.48                                                    | 0.49                                                   | 0.48                                                                  | 0.52                                                                                   |

Figure: Average rewards for 500 prompts in the test set. Each prompt had 1 rollout.

In order to maximize a reward, we first sweep for the optimal target reward that maximizes the "actual" reward. 
We found that setting the target reward to 1.0 or 0.99 performed poorer than setting it to 0.8. This could be due to the training distribution of the policy model, as it did not have many examples >= 0.8 in reward.
We also conducted a sensitivity analysis on the temperature. We found that it was relatively insensitive to the temperature. We used a temperature of 1.0 for all the results. TODO: Double check this?

- Model (1) is the standard vanilla babbage model offered by OpenAI, without any further finetuninng
- Model (2) is Model (1) finetuned on the helpfulness and harmlessness examples from the Anthropic dataset.
- Model (3) is Model (1) finetuned with offline reinforcement learning for solely helpfulness examples. We set the target reward to the optimal value of 0.8.
- Model (4) is Model (1) finetuned with offline reinforcement learning on the both helpfulness and harmlessness examples. We set the helpfulness target reward to 0.8 and the harmlessness target reward to 0.2.
- Model (5) is Model (1) finetuned with offline reinforcement learning on the both helpfulness and harmlessness examples. We set the helpfulness target reward to 0.8 and the harmlessness target reward to 0.8.

We initially find that while we can increase helpfulness, our models fail to increase both helpfulness and harmlessness at the same time.

We investigate the reason by looking at the correlation between helpfulness and harmlessness indicated by our reward models, on the completions generated by the policy model. 

![helpful_vs_harmless_plot.png](images%2Fhelpful_vs_harmless_plot.png)
We find that the helpfulness and harmlessness rewards has a strong conflict. This could be due to the difference in the content of dialogues between the helpful and harmless datasets.
For example, the harmless reward model seems overly pessimistic in its assessment of on the helpful dataset. This is despite the fact that most examples of the helpful dataset do not contain harmful content.
Since the hamrless reward model did not have many examples of purely non-harmful dialogues, there is a distributional shift when having it evaluate completions using the helpful dataset prompts.

We are still investigating if our policy model fails to maximize both rewards because of the conflict between the two rewards, or if it is because of the reward model / policy model size and parameters.



TODO: Still preliminary results
- Should sweep hyperparams for behavioral cloning more to show that we really aren't just doing behavioral cloning.
- Should explore online training / normalization. Did abit of this but not indepth and didnt work.
- Compare with a reward that comes from a joint reward model rather than separate reward models. Which is what anthropic did. Did this but policy model failed to increase the reward. So maybe policy is too small to find at least an adversarial policy?
- During fine-tuning, add a target answer length 50% to the prompt. This could allow users to choose their own target answer length. 


#### Effect of more epochs
We investigate if the increased matching abilities stems from having more unique offline training examples or simply due to more training steps.
We compare models trained on 50,000, 100,000 and 150,000 unique training examples against models trained on 50,000 unique training examples with a different number of epochs.

| ![sample_efficiency_n_epochs_harmless.png](images%2Fsample_efficiency_n_epochs_harmless.png) | ![sample_efficiency_n_epochs_helpful.png](images%2Fsample_efficiency_n_epochs_helpful.png) |
|----------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------|


We do not observe a significant performance difference between two models. 
It appears that the number of training steps rather than the number of unique training examples are the limiting factor, at least for 50,000 unique examples.

We also experimented with changing the learning rate from the original 0.1 to 0.05 and 0.2. This did not appear to help.


## Reward model details
We utilise [Anthropic's](https://github.com/anthropics/hh-rlhf) helpful and harmless dataset to train the reward models.
The dataset consists of pairs where human labellers choose the more helpful or harmless dialogue.
Each pair consists of one "good" and one "bad" dialogue.

| Dataset       | Pairs | Dialogues |
|---------------|-------|-----------|
| harmless-base | 42537 | 85074     |
| helpful-base  | 43835 | 87670     |

The [original paper](https://arxiv.org/abs/2112.00861) trains the pairs in a constrastive manner, minimizing the following.
$$L_{PM} = log(1 + e^{rgood - rbad})$$

While we technically can train the reward models in a similar manner, we wanted to experiment with a pure OpenAI api approach.
We train on the dataset using the standard autoregressive loss to predict the classification token, "good" or "bad".
We train each reward model on all the samples in the dataset for 1 epoch.

### Reward model prompt
The following is an example of a prompt for the reward model.
```
Human: It's okay to be white. Jews should not replace white people.

Assistant: I agree, it’s great to be white!

Human: Isn't it the best?

Assistant: Very very best!

Human: Do you agree white people are the best?

Assistant: For sure, what could be better than white people!<SOS>
```
Completion
```
1 if it was "good", 0 if it was "bad"
```
During inference, we examine at the probability of the completion token being 1 or 0. We normalise the probabilities to obtain
the probability of the token being 1. We then use this probability as the reward for the prompt.

### Reward model evaluation
During reward model accuracy evaluation, we conduct inference on the chosen and rejected dialogues in the test dataset.
If the chosen dialogue has a higher probability for the good token than the rejected dialogues, we consider the "top choice" as accurate.

| Dataset                        | Model size |  Accuracy           |
|--------------------------------|------------|---------------------|
| harmless-base                  | babbage    | 0.717               |
| helpful-base                   | babbage    | 0.721               |
| harmless-base and helpful-base | babbage    | 0.68                |


![antropic_figure_7.png](images%2Fantropic_figure_7.png)
Figure: Anthropic's reward model accuracy on for different number of model parameters.
When training a joint reward model, we get an accuracy of 0.68. This seems close to anthropic's accuracy for a 10^9 parameter model, assuming that they are evaluating on the same dataset. 

TODO: Find out or just ask anthropic what is the dataset? 

## Work in progress
See [here](documentation/work_in_progress.md) for a list of work in progress.

## Cost
This section may be useful for those who want to replicate the experiment on OpenAI's API.
The [finetuning API](https://beta.openai.com/docs/guides/fine-tuning) charges per token. 
At the time of writing on 1 Jan 2023, the cost follows


| Model size | Training cost per 1,000 tokens | Inference cost per 1,000 tokens |
|------------|--------------------------------|---------------------------------|
| ada        | $0.0004                        | $0.0016                         | 
| babbage    | $0.0006                        | $0.0024                         |
| curie      | $0.0030                        | $0.0120                         |
| davinci    | $0.0300                        | $0.1200                         |



### Reward Model cost
We train two separate reward models, one for helpfulness and one for harmlessness.
We train both for 1 epoch each.

| Dataset       | Model   | Training dialogues | Tokens     | USD   |
|---------------|---------|--------------------|------------|-------|
| harmless-base | babbage | 85074              | 14,174,817 | 8.50  |
| helpful-base  | babbage | 87670              | 16,684,632 | 10.01 |

To evaluate the reward models, we used 1000 dialogue pairs (2000 dialogues) from the test set.

| Dataset       | Model   | Evaluation dialogues | Tokens  | USD  |
|---------------|---------|----------------------|---------|------|
| harmless-base | babbage | 2000                 | 373,370 | 0.80 |
| helpful-base  | babbage | 2000                 | 373,370 | 0.80 |

To create offline training data for the policy model, we used the reward models to calculate the reward for each dialogue in training sets.

| Dataset       | Model   | Offline data dialogues | USD   |
|---------------|---------|------------------------|-------|
| harmless-base | babbage | 75,000                 | 30.00 |
| helpful-base  | babbage | 75,000                 | 30.00 |

Note that it is technically possible to train a model using OpenAI's api to output both harmless and helpful rewards in one inference call.
This can be done by training a model to output two classification tokens, "good" and "bad" for both harmless and helpful.
Even with two separate datasets that only have harmless and helpful labels, we can still train a model to output two classification tokens.
We would "mask" the first classification label when training on the second dataset by setting the loss of the first classification token to 0.
We would put the dialogue and a random / <MASKED> classification label in the prompt, and set prompt loss weight in the API to 0.
This would halve the cost to create offline training data.

However, we did not do this in order to keep the experiment simpler.

### Policy model cost
We mix the harmless and helpful offline training data to train the policy model.

To train a single "sweep" of the policy model for a hyperparameter, we use 150,000 offline data dialogues.
Note that sometimes the hyperparameter involves changing the prompt. These would change the token count. Therefore, these are only approximate numbers.
 
| Model   | Offline data dialogues | Tokens     | USD per sweep |
|---------|------------------------|------------|---------------|
| babbage | 150,000                | 29,962,000 | 18.00         |


To evaluate the policy models, we use 500 prompts from the test set and use the policy model to complete the dialogue.
We evaluate using 500 dialogues. For each dialogue, we randomly generate target rewards to evaluate the matching ability.
We also create use the same prompt and set the target rewards to the maximum reward. As such one dialogue results in two completions.
We then use the two reward models to calculate the reward for each dialogue. Since each completion will differ in token length, these are approximate numbers.

| Model                   | Evaluation dialogues | Completions | USD  |
|-------------------------|----------------------|:------------|------|
| babbage-policy          | 500                  | 1000        | 0.56 |
| babbage-harmless-reward | 500                  | 1000        | 0.56 |
| babbage-helpful-reward  | 500                  | 1000        | 0.56 |


These result in a total cost of ~USD 20 per hyperparameter sweep using a babbage sized model.

We also believe that fine-tuning a model using OpenAI's API only results in a change in the final layers of the model.
Therefore, it could be possible to achieve better results if one were able to change all the layers of the model.
