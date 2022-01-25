---
title: "Support Vector Machines - Passerini"
author: "Mattia Carolo - @Carolino96"
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

# Support Vector Machines

Support Vector Machine, SVM from now on, are linear classifiers that separate using a **large margin classifier** which solution depends only on a small subset of the traing examples called **support vector**. It's very important to note that it has a sound generalization theory (not to study) and they can be easily extended to non linear separation retaining the separation properties thanks to *kernel machines*.

# Maximum margin Classifier

Let's try to formalize the margin. We already know that $yf(x)$ is the confidence on the correct prediction, if negative the prediction is wrong otherwise if positive correct and the value is the confidence on the prediction. Now suppose we have a classifier that correctly separates with no training errors. If this is the case the minimum value among the training examples is called *confidence margin* and it's written like
$$
\rho = \min_{(\bold x, y)\in D}yf(\bold x)
$$

Since it depends on $w$ we can compute the distance from the minimal distance to our classifier and it's called **geometric margin** which is formalized like 
$$
\frac{\rho}{\|\bold w\|} = \min_{(\bold x, y)\in D}\frac{yf(\bold x)}{\|\bold w\|}
$$

Ideally we want to maximize the last formula in order to get $w$ in order to maximize the margin. However if we put in an optimization problem we have actually one degree of freedom that is being removed. Suppose we have a solution where
$$
\mathbf{w}^T\mathbf{x} + w_0 = 0
$$

now if we want to characterize further the plane we can, for example, multiply the terms with an $\alpha \neq 0$ and still we will return to a formula that look like before since we can incorporate the $\alpha$ in our formalization. This is because there is an infinite number of equivalent formulation for the same hyperplane even with different parameters.

We can counter this problem through the introduction of the *canonical hyperplane* in which we set the constraint that $\rho$ must be equal to a number given a priori (in our case we take 1) in order to get:
$$
\rho = \min_{(\bold x, y)\in D}yf(\bold x) =1
$$
and it's geometric margin will be $\displaystyle \frac{\rho}{\|\bold w\|} =\frac{1}{\|\bold w\|}$