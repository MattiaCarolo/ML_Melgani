# Minimum Risk Theory

With the precednt criteria we can meet an issue. Precedent criteria never took in account the cost of an erroneus classsification.
The cost is the cost of a given a decision represented by a class.

So to compute all the cost function based on a particular action $R$ we need to create a matrix $RxC$ in which will contain all the possible action paired to the relative cost function.
This relation is given by:

$$ \lambda _ij(\alpha _i | \omega _j) $$

where:

- $\alpha$ is the action taken
- $\omega$ is the class to which we consider the observation belongs
- $\lambda$ is the cost incurred when we take an action $\alpha$ given the class $\omega$

For each pattern $x_i$ we can introduce the **conditional risk** which is the expected loss of taking action $\alpha$ which (check online) given an observation $x_i$

We can compute even the **overall risk** where we compute the risk of taking a given action considering all the feature space. Since we are computing actions across all the feature space we achieve the best performace called **Bayes risk** where all the risk correlated to the action is minimized.

# Discriminant Function