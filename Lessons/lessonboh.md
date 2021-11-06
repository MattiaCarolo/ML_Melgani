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


date 25/10

# Decision trees

Till now we only spke about how to classify something having in mind that the feature vector was containing only numeric values so at the end the classification was just a partitioning of the feature space. Than whenever we had a sample to classify we just assigned it to the feature space. A tool used in this was the computing of the distance in order to assign a value to a particular feature space since it can quantify how much a sample is similar to a class.

This tho was only with numerical data. If we introduce nominal data like "high,medium,low" or the color of a particolar object we can't quantify it so we are considering non-numeric data.

To tackle this kinf of problem we need to take a completely different approach to this situation in where the solution was taken by modeling the human behaviour in recognizing something.
This solution is found with decision trees where for every non terminal node we get a question to identify the data. After we complete the sequence of questions on the terminal node we will get our answers. Links between nodes must be mutually distinct and exhaustive

## CART

To implement this kind of solution we utilize the paradigms given by the CART framework which can be implemented to make different decision trees. This framework raises different aspects:

### Split Number

For split we intend each decision outcome which splits the subset of training data

> **NB** the root node splits all

### Test Selection

Each node is a test that will define our decision boundary for our solution.

### Node Impurity

In order to build a ncie tree we need to compute the purity of the test in order to understand how we descend through our tree and consequentively the order of our decision boundary. In order to decrease the impurity we could choose the test which reduces more the impurity but it could result only in a local optimization 

### Twoing Criterion

Used for multiclass problem when we want to use binary tree creation

### Stopped splitting

At each iteration where we reduce the impurity we risk to overfit the data if we compute it until every leaf results in the lowest impuruty or otherwise if we stop early the error on classification could be too high and hence performance may suffer.

### Pruning

With this approach we manage a fully grown tree where we merge different cells together. The criteria to merge is if i merge 2 cells the resulting impurity must be lower than a certain threshold.

With a lot of treaining sample the tree could result in a huge set so pruning it can lead to extendet times of operations

### Leaf node Label Assignment

After all this we need to label cells with the dominant class we find inside our cell

### Generalization Error 

27/10

#1

COnsidering a machine xi -> y and assuming

A machine is defined by a set of possible mappings x related to the function with some parameters $\alpha$ 

$$ R(\alpha) = \int{R(\alpha|{x,y})p({x,y}){dxdy}} $$

where we can compute our loss function 

$$R(\alpha|{x,y}) \Rightarrow $$

### Accuracy Evaluation  

### Binary classifier

### ROc Curve

Independetly from the choice of our threshold we still need a measure to quantify the goodness of a estimation