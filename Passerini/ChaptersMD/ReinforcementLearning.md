---
title: "Reinforcement Learning - Passerini"
header-includes:
   - \usepackage{cancel}
   - \usepackage{tikz}
   - \usepackage{amsmath}
   - \usepackage{xcolor}
   - \usepackage{neuralnetwork}
   - \newcommand{\indep}{\perp \!\!\! \perp}
output:
    pdf_document
---

# Reinforcement Learning

It’s a learning setting, where the learner is an **Agent** that can perform a set of **actions** $A$ depending on its state in a set of **states** $S$ and the environment. In performing action a in state $s$, the learner receive an immediate **reward** $r(s,a)$ and we'd want to maximize the rewards.

In some states, some actions could be _not possible/valid_.

The task is to **learn a policy** allowing the agent to choose for each state $s$ the action $a$ **maximizing the overall reward**, including future moves which gives *delayed rewards*.

To deal with this delayed reward problem, the agent has to trade-off _exploitation_ and _exploration_:

- **exploitation** is action it knows give some rewards
- **exploration** experience _alternative_ that could end in bigger reward

This involves also a task called **credit assignment**, to understand which move was responsible for a positive or negative reward. 

> like playing a game and see the win or lose to understand if it was a bad move and in case don’t repeat it

## Sequential Decision Making