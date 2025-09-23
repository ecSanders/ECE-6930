## Tic-Tac-Toe Report

#### Q: How is the agent and environment setup?

> A:

>$S$ : A 1 x 9 char array with 'X', 'O', or ' ' filling it out. 

>$A$ : An array index ranging from 0-8 to indicate where an agent attemps to play.

>$P(s'|s,a)$: Given a state and action how likely is the outputted state.

>$R(s',a,s)$: Reward given at the end of a game, possible outcomes are 1, 0, and -1. (Win, draw, and tie)

#### Q:Did $V(s_{empty})$ or win-rate improve?

> A: They both did. The results are as given:

Estimating V under random policy...

V_random(empty) ≈ 0.2962

Evaluating random vs random:

Random policy as X vs random O: win=0.586, draw=0.130, loss=0.283

Evaluating improved policy:

Improved policy as X vs random O: win=0.928, draw=0.035, loss=0.037

> The win-rate improved as a result of the policy being improved. The policy was improved by performing a model-based lookahead. For each action, we computed the expected value using both the transition probabilities and rewards. Afterwards, we perform a greedy action by taking the argmax. This lookahead and update ultimately improved our policy which in turn improved or win-ratio. 