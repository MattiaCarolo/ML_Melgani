# Lesson 3-4

## Estimation basics

In estimation theory, stochastic signlas can be subdivided in three categories:

- **Noisy Deterministic signals**
  - The information source is completely known. Noise interference can be read only during the transmission or acquisition phase
- **Noisy Parametric signals**
  - The information source is only partially known. Observations allow estimating the random parameters associated to their relative signals
- **Noisy Random signals**
  - The signal is completely unknown. In this case every estimation relies only on the observation given since there is no other knowledge to leverage.

For the course will only be treated the second and third category since they are most common in most real world applications.

## PDF Estimation

**PDF** stands

### Parametric estimation

In parametric estimation we assume we know the shape of the probability function of the parameters. These parameters are stored in a vector called $\theta$.
Plus we have a vetor $X$ which stores the training set of our model.

$p(X|\theta) = \prod_{k=1}^Np(x_k|\theta)$

Using this likelihood function we can understand how much our training set fits our training model.
Maximizing our likelihood function permits to understand which training set fits better.

### Estimation Procedures

#### Bayesian estimation

teta its a vector of random variables where we assume a prior knowledge of the distribution of the single variables.

### Estimation Goodness (voltimeter example)

Through a battery and a voltimeter we want to measure the voltage of the battery.
Connecting the battery and mesuring it we take an estimate but probably we will not have the true voltage of the battery but there will be a **bias**. This bias can be compensated knowing the bias and removing it from the value displayed by the voltimeter.
With a different voltimeter than we can have different values registered by the voltimeter. This problem is called **uncertainity** fow which we need to compute the **Variance** which is the measure of our uncertainity.

The estimate of the vector of the parameters depends on the observation vector X represented like $\theta = \theta (X)$ so our estimate vector is a random vector.
Our estimation error $\epsilon$ where 
$\epsilon = \theta - \theta = \epsilon(X,\theta) = [\theta_i - \theta_i, \forall i]$
We need for an ideal estimator two things.
To be unbiased and not variable. to be unbiased we check simply if the error is equals to zero so $\theta$ computed must be the same of our model.
For the Variance

#### Cramer-Rao Bound

We need our variance to be greater than or equal to this bound.

### Maximum Likelihood Estimation

taking as example some gaussian models in which we know the variance but not the mean. To maximize the likelihood function we need to maximize the function related or to minimize/maximize the inner function.

In a finite number of training examples, if it exist an efficient estimate and the ML estimation is unbiased, than the ML will be the efficient estimate

### Model Selection

Even if we have a complete deterministic environment still, during our processing of the information, we introduce some noise in the information processed due to physical and even mathematical reasons (kernelling, bad sensibility etc.etc.)
The choice so it's entirely made by the supervisor heuristically. Usually we take a model which fits on our observation

### Gaussian Model

This model is widespread for a mathematical reason called **central limit theorem** which tells that if the sum of all variables has a finite variance than the result will still be gaussian

In a 2D Gaussian PDF we can look at our distribution like a bell shaped distribution. We can cut the bell in parallel planes which will form our isolevels and are shaped like an ellipses.

complete correlation x1 = x2 which means that we only have linear proportion between the two features. Increasing the $\Delta$ between the fetures results in a more big ellipses.