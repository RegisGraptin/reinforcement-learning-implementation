
# DQN (Deep Q-network) approach

Given an environment, we want to create a model that will take accurately actions in order to maximize the future reward. The DQN model combine two approaches, the reinforcement learning and the deep learning fields.

The usage of deep learning is interesting because, it force the model to create an efficient representations of the environment from high-dimensional sensory inputs. In the following example implemented, we only have 4 environment's input. But, this model can be used in a more complex representation, as it have been used in the [1](Mnih et al 2015) approach where they used convolution networks to represent environments. 

## Description 

The goal of the agent is to select actions in a fashion that maximize cumulative future reward. 
Here, use deep learning network to approximate the optimal action-value function.

$ Q^*(s,a) = \max{\pi} \mathbb{E}[r_t + γr_{t+1} + γ^2r_{t+2} + ... | s_t = s, a_t = a, \pi] $

RL is known to unstability or divergence when nonlinear functions are used as an approximator (the case for neural network). This is due to small update of Q may significantly change the policy and therefore the data distribution and the correlation between the action-value and the target value. 

The solution proposed in (Mnih et al 2015) is to use two model:
- A policy model: that we trained and apply on our training environement. 
- A target model: that is used as a referent. 


The process is decomposed in two parts:
- "Experience replay": in order to train the policy model, we will take the training data in a **randomize way**. The data will correspond to the action taken by the policy model at a given state leading to a reward and the next state.

- "Iterative update": to improve our approach, we also need to improve the target model. To update it, we will adjust the weight of this model iteratively (example: each 100 normal training). This will allow us to adjusts the action-values (Q). Else, the target model are held fixed between individual updates of the policy net.



We parameterize an approximate value function Q(s, a $ θ_i $)
=> $ θ_i $ : weights of the Q-network at iteration i


For experience replay, we store the agent's experiences as $ e_t = (s_t, a_t, r_t, s_{t+1}) $ at each time step t in a data set $ D_t = {e_1, ..., e_t} $.

During learning we apply Q-learning updates, on samples (or minibacthes) of experiences e~U(D). We then, at iteration i, use the following loss:

$ L_i(θ_i) = \mathbb{E}_{(s,a,r,s') ~ U(D)} [ (r + γ \max_{a'}Q(s',a';θ_i^-) - Q(s, a; θ_i))^2 ] $


- $ θ_{i} $  parameters of the Q-network at iteration i.
- $ θ_{i}^{-} $ : are only updated every C steps and are held fixed between individual updates.



> Reference: 
- [1] Mnih, V., Kavukcuoglu, K., Silver, D. et al. ["Human Level Control Through Deep Reinforcement Learning"](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf). Nature 518, 529–533 (2015). https://doi.org/10.1038/nature14236

## Additionnal notes

- Do not use softmax function for the output function of the model.

> Softmax output is quite popular for representing the policy network/function for discrete actions. But for Q learning, we are learning an approximate estimate of the Q value for each state action pair, which is a regression problem.
>
> -- [<cite>Gary Wang</cite>](https://www.quora.com/Why-does-DQN-use-a-linear-activation-function-in-the-final-layer-rather-than-more-common-functions-such-as-Relu-or-softmax)