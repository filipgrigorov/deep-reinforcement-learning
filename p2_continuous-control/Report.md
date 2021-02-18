[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/43851646-d899bf20-9b00-11e8-858c-29b5c2c94ccc.png "Crawler"


# Project 2: Continuous Control

### Learning algorithm

For this project and according to specification, the **DDPG** (Deep Deterministic Policy Gradient) algorithm has been implemented, using the multiple agent environment. 

Even though, it is not quite actor-critic algorithm in the sense that the critic learns to approximate the maxiing function of the state-value estimate, instead of a baseline against which the actor is benchmarked.

Basically, the actor learns to optimize the policy with the maximum state-action estimate for a given state, thus, following a deterministic policy (again, maximizing the estimate for a state-action value):

$$argmax_{\theta_{\mu}} = E_{S \sim D}[Q_{\theta_{Q}}(s, \mu_{\theta}(s; \theta_{\mu}); \theta_{Q})]$$

And the critic, on the other hand, learns the optimal $Q_{\theta_{Q}}^{*}(s, a; \theta_{Q})$ where $a =  \mu_{\theta}(s; \theta_{\mu})$ and $\mu(s) = argmax_{a}Q(s, a)$, theoretically. So, it tries to optimize $$argmin_{\theta_{Q}} E_{s,a \sim D}[(Q(s, a; \theta_{Q}) - y)^{2}]$$ where $y = r(s, a) + \gamma Q(s^{'}, \mu(s^{'}); \theta_{Q})$.

The algorithm uses soft-update for stability (as in the DQN algorithm):
$$\theta_{\mu}^{'} = \tau * \theta_{\mu} + (1 - \tau) * \theta_{\mu}^{'}$$
$$\theta_{Q}^{'} = \tau * \theta_{Q} + (1 - \tau) * \theta_{Q}^{'}$$
where $\theta^{'}$ is the target.

Hyper-parameters:

seed = 1

batch_size = 128

memory_size =1e5

$\gamma = 0.99

actor_lr = 1e-3

critic_lr = 1e-3
