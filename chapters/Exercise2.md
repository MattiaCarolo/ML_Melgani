## Exercise on ML Estimation

Let's suppose we have a poisson distibution which is formalized like
$$
p(x_i|\lambda) = \frac{\lambda^{x_i}-e^{-\lambda}}{x_i!}
$$

and a set of $N$ training samples $X = \{x_1,x_2...,x_n\}$ and we need to estimate $\lambda$

To resolve this type of exercise we need before to define the likelihood funciton than to maximize it.

### Step 1

Compute likelihhod and $\log$ likelihood function

$$
P(X|\lambda) = \prod_{i=1}^{N}p(x_i|\lambda)
$$

$$
\ln P(X|\lambda) = \sum_{i=1}^{N} \ln p(x_i|\lambda)
$$

$$
\sum_{i=1}^{N} \ln p(x_i|\lambda) \Rightarrow \sum_{i=1}^{N}[x_i \ln\lambda - \lambda - \ln x_i!]
$$

### Step 3

Maximize likelihood

$$
\frac{\partial \ln p(x_i|\lambda)}{\partial \lambda} = 
\sum_{i=1}^{N} \ln p(x_i|\lambda) \Rightarrow \sum_{i=1}^{N} [\frac{x_i}{\lambda}-1] = 0
$$

$$
\frac{1}{\lambda} \sum_{i=1}^{N} x_i = N \Rightarrow \hat\lambda = \frac{1}{N} \sum_{i=1}^{N} x_i
$$

## Exercise on Bayesian estimation

Let's suppose we have a poisson distibution which is formalized like