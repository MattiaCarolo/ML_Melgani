# Chapter 6

## Support Vector Machine

Linear classifier which maximizes the separation margin between the two classes. The solution depends on a small subset of training example where the idea is the result of the training depends on few critical examples.
The generalization error is strictly correlated to the margin.
A perk of SVM is that it can be estended to nonlinear separation (*kernel machines*)

### Maximum Margin Classifier

Given a training set $D$ a classifier confidence margin

$$ \rho = min_{(\bold{X},y) \in D } yf(\bold{x})$$

and it is the minimal confidence margin among the training examples which is used to predict the true label

$$ \frac{\rho}{||\bold{w}||} = min_{(\bold{X},y) \in D } \frac{yf(\bold{x})}{||\bold{w}||}$$

In a canonical hyperplane there is an infinite number of equivalent formulation to represent it and this means that the separating 