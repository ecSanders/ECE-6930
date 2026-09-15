Part D: Result Comparison
Write a short discussion addressing:

Why Double DQN is usually more stable.

What overestimation bias is, and how you see it (or don’t see it) in your results.

Any differences you noticed when changing hyperparameters (episodes, max steps).

Deliverable: PDF or Markdown summary.

## DDQN: Addressing Maximization Bias

Because the network's Q-value estimates are imperfect and noisy, the “max” tends to pick actions whose Q-values are overestimated due to random error, not because they're better.
Over time, this accumulates, leading to unstable learning, divergence, or oscillating Q-values. We see the instability in our previous chart as noise.


## DDQN: Hyperparameter Tuning

The idea is that as you increase episodes and moderate step sizes you get smoother learning.