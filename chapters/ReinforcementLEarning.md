# Reinforcemnet Learning

To the learner is provided a set of possible states $S$ and for each state a set of possible action $A$ moving to the next state. When an action $a$ is performed that results in $s$ to the learner is provided an immediate reward $r(s,a)$

The task odf reinforcement learning is to learn a policy in order to maximize all possible reward taking into account even future moves. FOr this the learner has to deal with delayed rewards caused by future actions while considering the tradeoff between *exploitation* and *exploration*

## Decision Utilities



AN agent needs to take a sequence of decisions in which it should maximize some utility function.
> N.B. there is an uncertainity in the result of a decision

### Markov Decision Process

Given:

- a set of **states** $S$ in which the agent can be at each time instant
- a (possibly empty) set of terminal states $S_G \subset S$
- a set of **actions** $A$ the agent can make
- a **transition** model providing the  probability of going to $a$ state $s_0$ with action $a$ from state $s$
$$
P(s'|s,a){\spades} \space s,s'\in S,\space a \in A
$$
- a **reward** $R(s,a,s')$ for making action $a$ in state $s$ and reaching state $s'$

### Defining Utilities

Utilities are dwfined over the **environment history** which is a sequence of states we traverse if we take a certain order of action.

Over this we assume an **infinite horizon** which in principle it means that there is no time constraint on the maximum time.
We assume even that there are **stationary preferences** which means that if an history is preferred respect to another at a certain time $t$ the same should be at time $t'$ provided they start from the same state

### Taking Decisions

a **policy** $\pi$ is a full specification of what actions to take at each state while an **optimal policy** is a policy which maximizes the expected utility. Moreover for infinite horizons, optimal policies are **stationary**

The **expected utility** of a policy is the utility of an environment history, taken in expectation over all possible
histories generated with that policy.
