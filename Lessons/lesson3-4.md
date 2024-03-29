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

Before talking about the estimation problem though we need some notations in order to get a better formalization.

Let $x = \{ x_1,x_2,....x_n \}$ be a vector of $n$ features of an unknown pdf (e.g. the features extracted from an image of a person). This vector in a N Dimensional feature space in which our vector will identify one point in the space.

Let $X = \{ X_1,X_2,....X_N \}$ be the set of $N$ samples we have which we will use to crete our model. These samples will be called *training samples* and will be the base to estimate our model.

## Parametric estimation

In parametric estimation we assume we know the shape of the probability function of the parameters. These parameters related to the model $p(x)$ are stored in a vector called $\theta$ where $\theta =(\theta_1,\theta_2,....,\theta_n)$ . And since our model is parametric (depends from certain parameters) his density won't be based only on the training itself so we underline this by using the notation $p(x|\theta)$

So assuming having a set of independent training samples $X$ we can introduce the **likelihood function**

### Likelihood Function
A likelihood function it's a source with different observations. Through this function we can quantify the matching between the set of training samples and the parameters of the model and it's identified as $P(X|\theta)$ where since 

$$ p(x_1,x_2 .... x_n | \theta) = P(X|\theta) $$

than 

$$ p(x_1,x_2 .... x_n | \theta) = p(x_1|\theta)*p(x_2|\theta)*......*p(x_n|\theta) $$

so we can formalize it like

$$ P(X|\theta) = \prod_{k=1}^Np(x_k|\theta )$$

This matching is given by the formula above where the joint probabilities between the sets is simply the product between the single probabilities depending on our parameters.
Through this function we can understand how much our training set fits our training model resulting in an understanding of which set is better.

> #### Exercise on Likelihood Function

Let's assume we have a set of training samples represented from the red dots on the image below. Since we are dealing with a parametric model we know the model which for this example will be gaussian so our $p(x|\theta)$ will be $p(x|\mu ,\sigma^2)$ which will follow a normal density $N(\mu,\sigma^2)$.

For the sake of the exercise we consider a $\sigma$ of 1 in order to keep a gaussian with unitary varaince but still we don't know where to put it since $\mu$ which is a continuos value can be placed everywhere in our $x$ axis so it has infinite possiblities so we need values for which the model referring to our $\mu$ covers the training set.

|![Training set](../Img/Chapter2/TrainingSets.png "Training set and models")|
|:--:|
|**Training Set and Candidate Models**|

To quantify how much a model fits the training set we need to compute the likelihood function which in this case wil be represented from the gaussian function 

$$ \frac{1}{\sigma \sqrt{2\pi}}\exp(-\frac{(x-\mu)^2}{2\sigma^2}) $$

which with $\sigma = 1$ is

$$ \frac{1}{\sqrt{2\pi}}\exp(-\frac{(x-\mu)^2}{2}) $$

so $\mu$ will be our unknown feature to find.
To compute it we need to compute every training sample over our $\mu$ so 

$$ p(x_1,x_2,... x_n | \mu) (A) = p(x_1|\mu)*p(x_2|\mu)*....*p(x_n|\mu) (B)$$ (1)

$$ (A) = \frac{1}{\sqrt{2\pi}}\exp(-\frac{(x_1-\mu)^2}{2}) * \frac{1}{\sqrt{2\pi}}\exp(-\frac{(x_2-\mu)^2}{2}) *.... * \frac{1}{\sqrt{2\pi}}\exp(-\frac{(x_n-\mu)^2}{2})$$ (2)

$$ (A) = \frac{1}{\sqrt{2\pi}} \prod_{i=1}^{N} \exp(-\frac{(x_i-\mu)^2}{2})$$ (3)

where for $x_i$ we mean the single sample iterated over our likelihood function.

> **NB** Care that in this case we can do the product beacuse the product of a gaussian is still a gaussian. For other types of model we need to take into care other forms to compute the likelihood

After the computation we will have as a result a gaussian with the mean calculated from the training sets with our $\hat{\theta}$ as the computed mean. This point will be the maximum agreement between the training sets and the model.

|![Training sets](../Img/Chapter2/NarrowMeanLikelihood.png "Comparison between Uninformed searchs")|
|:--:|
|**Computed Gaussian**|

Still when we work with Maximum Likelihood Estimation in general it's preferred to work with a log function in order to get rid of the exponential term and this will make computation more easier. With this method we don't lose the generality since our $\hat{\theta}$ will remain the same because the logarith is a monotonic function

|![Training sets](../Img/Chapter2/LogLikelihood.png "Comparison between Uninformed searchs")|
|:--:|
|**Training sets**|

## Estimation Procedures

There are two main procedures for parametric estimation which are:

- *Maximum Likelihood Estimation* $\Rightarrow$ we look at $\theta$ as a vector of parameters as a vector of unknown constants.
- *Bayesian Estimation* $\Rightarrow$ $\theta$ its a vector of random variables where we assume a prior knowledge of the distribution of the single variables. This prior density will contain the knowledge of the experts and will be used to get the posterior density (we'll talk later about it)

## Estimation Goodness (voltimeter example)

Through a battery and a voltimeter we want to measure the voltage of the battery.
Connecting the battery and mesuring it we take an estimate but probably we will not have the true voltage of the battery but there will be a **bias**. This bias can be compensated knowing the bias and removing it from the value displayed by the voltimeter.
With a different voltimeter than we can have different values registered by the voltimeter. This problem is called **uncertainity** fow which we need to compute the **Variance** which is the measure of our uncertainity.

The estimate of the vector of the parameters depends on the observation vector X represented like $\theta = \theta (X)$ so our estimate vector is a random vector.
Our estimation error $\epsilon$ where 
$\epsilon = \hat{ \theta } - \theta = \epsilon(X,\theta) = [\hat{\theta_i} - \theta_i, \forall i]$
We need for an ideal estimator two things.
To be unbiased and having no variance. 

An estimator is called **unbiased** when the expected value from $\epsilon$ becomes 0 or in other words:

$$ E\{ \epsilon \} = 0 $$

where if there is no error it means that our estimate coincides with the model so

$$ E\{ \hat{\theta} \} = \theta $$

to be unbiased we check simply if the error is equals to zero so $\theta$ computed must be the same of our model.
For the Variance we define id as the variance of our $\epsilon_i$ where

$$ var\{ \epsilon \} = E\{(\hat{\theta_i} - \theta_i)^2\} \space {where} \space \theta_i(i=1,2,3....,r) $$

but to assess whether our variance is good or not we need to define a lower bound called the **Cramer-Rao Bound**

### Cramer-Rao Bound

We need our variance to be greater than or equal to this bound. The more our variance comes closer to this lower bound the more our estimators will be unbiased. Formalized we can express it like:

$$ var \{ \epsilon_i\} \ge [I^{-1}(\theta)]_{ii} {with} \space i=1,2,3...r $$

where $I(\theta)$ is the **Fisher information matrix** which is defined as a matrix where each element is computed like

$$ [I(\theta)]_{ij} = E\{ \frac{ \partial \ln [p(X|\theta)]}{\partial \theta_i} \cdot \frac{ \partial \ln [p(X|\theta)]}{\partial \theta_i}\} $$

where as we can see it's the derivative of the log likelihood function

Problem is that often estimates form real problems are obtained with biased and inefficient estimators so in order to judge better the estimation of our estimator we need large set of observations. This means that an estimator to be good must have good asymptotic properties.
It can be asymptotically unbiased if 

$$ \lim_{N\rightarrow +\infty} E\{\epsilon\} = 0 \Rightarrow \lim_{N\rightarrow +\infty} E\{\hat{\theta\}} = \theta  $$

while to be asymptotically efficient if 

$$ \lim_{N\rightarrow +\infty} \frac{{var}\{\epsilon_i\}}{[I^{-1}(\theta)]_{ii}} = 1 \space {with} \space i=1,2....,r $$

So an estimate to be considered efficient need to be **consistent**. Consistent means that it needs to converge to the true value when the number $N$ of samples tend to infinite. The necessary condition for this is that the estimate is asymptotivally unbiased and with variance converging to zero when $N \rightarrow +\infty$

## Maximum Likelihood Estimation

The maximum likelihood estimate (ML) of $\theta$ is the estimator that maximizes the argument $\theta$ so
$$ \hat{\theta} = \arg \max_{\theta} p(X|\theta) $$

|![Training sets](../Img/Chapter2/UnivariateGaussian.png "Comparison between Uninformed searchs")|
|:--:|
|**Mono Dimensional Gaussian**|

Taking as example the image above with some gaussian models in which we know the variance but not the mean and in this estiamation we are given only one sample. To maximize the likelihood function we need to maximize the function related or to minimize/maximize the inner function of the model. In order to do this we will compute our likelihood function over and over until we will reach the maximum argument

In a finite number of training examples, if it exist an efficient estimate and the ML estimation is unbiased, than the ML will be the efficient estimate

> ### Properties
> Even if in reality there is not an efficient estimate the ML estimate exhibits good aymptotic properties since it is:
> - asymptotically unbiased
> - asymptotically efficient
> - consistent

### Statistical Model Selection

Even if we have a complete deterministic environment still, during our processing of the information, we introduce some noise in the information processed due to physical and even mathematical reasons (kernelling, bad sensibility etc.etc.)
The choice so it's entirely made by the supervisor heuristically. Usually we take a model which fits on our observation

### Gaussian Model

This model is widespread for a mathematical reason called **central limit theorem** which tells that if the sum of all variables has a finite variance than the result will still be gaussian

In a 2D Gaussian PDF we can look at our distribution like a bell shaped distribution. We can cut the bell in parallel planes which will form our isolevels and are shaped like an ellipses.

complete correlation x1 = x2 which means that we only have linear proportion between the two features. Increasing the $\Delta$ between the fetures results in a more big ellipses.