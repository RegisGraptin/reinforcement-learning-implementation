

## Note

- Do not use softmax function for the output function of the model.

> Softmax output is quite popular for representing the policy network/function for discrete actions. But for Q learning, we are learning an approximate estimate of the Q value for each state action pair, which is a regression problem.
>
> -- [<cite>Gary Wang</cite>](https://www.quora.com/Why-does-DQN-use-a-linear-activation-function-in-the-final-layer-rather-than-more-common-functions-such-as-Relu-or-softmax)