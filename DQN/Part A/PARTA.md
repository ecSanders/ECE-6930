Part A: Conceptual Foundation
Short Write-Up

Explain the process of DQN in your own words.

Specifically: (1) how it extends Q-learning, (2) the role of the target network, (3) the replay buffer, and (4) how the Bellman equation is approximated.

Compare this to the MDP formulation you did for Tic Tac Toe in Assignment 1.

Deliverable: PDF or Markdown summary.

## DQN: Extending Q-Learning

The DQN is an off-policy reinforcement learning network that compensates for the weaknesses of  Q-learning which are:
- The inability to handle large or complex state spaces 
- Learning from correlated experiences

DQN addresses these by:
- Using network weights to create a "Q-table" (not really a q-table but the weights are the learned values)
- Using a replay buffer to break correlation and use better data efficiency

## DQN: The Target Network

One of the challenges in a DQN is stability. If $\theta$ is updated, the target keep changing.

To fix that, DQN introduces a second network, the target network, $Q_{\theta-}$

- The main network $Q_\theta$ learns and updates every step.
- The target network $Q_{\theta-}$ used only to compute the target value $y$.
- Every few thousand steps, you copy the weights from the main network to the target network

This makes the target more stable and prevents feedback loops that cause divergence.

## DQN: The Replay Buffer

In Q-learning, updates occur after each new step, meaning the updates are based on consecutive states, and these are highly correlated.

Neural networks don’t learn well from correlated data...

To fix this DQN introduces a replay buffer.

At each learning step, DQN samples a random mini-batch from the replay buffer instead of using the most recent transition.

This breaks correlation between samples and improves efficiency by reusing old experiences multiple times.

## DQN: The Bellman Equation

The Bellman equation expresses how optimal Q-values relate recursively:

Since we can’t compute $Q^*$ exactly, DQN approximates it using the neural network.

For each sampled transition we compute the target value, compute the loss, and then use gradient descent to adjust the weights of the network to reduce the loss.

Over time, the network converges toward '$Q^*$', just like tabular Q-learning, but using the function approximation of a neural net.

## DQN: Tic-Tac-Toe

Honestly, the forumlation is still the same, but the implementation is wildly different. DQN's are still Q-learning just scaled up to use network weights instead of a table.