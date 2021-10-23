# Lesson 5

## Nonparametric Estimation

This kind of estimation becomes necessary when :

- There is no prior knowledge about the model 
- parametric model do not offer good approximation of the model (img 1)

### Basic concepts

let $x*$ be a sample of a region $R$ so that $x^* \in R$.
The probability $P_R$ = $\int _R p(x)dx = p(x^*)V$ which means the probability asoociated to the region R is its volume multiplied the density probability if $x^*$. Graphically we can see this as the samples belonging to R versus all the samples.
fig 2
In this :

- $k$ is the number of samples belonging to R
- $N$ total of samples

A consisstent estimate of $P_R$ can be achieved through the use of the relative frequency.
\[ 1) P_R = K/N \]
\[ 2) P_R = p(x^*)V_R \]
\[ 1 = 2 \Rightarrow p(x^*)V_R = K/N  \Rightarrow p(x^*) = \frac{k}{NV_R}\]

Depending on what parameter (K or V) we want to leverage we use a different method:

- **K Nearest Neighbour Method** has a fixed K and tries to compute the volume 

### K-NN Estimation

In this estimation the number K is alreay set and we need to set a shape for the cell volume with $x^*$ centered which is chosen beforehand.

The K-NN method consists on expanding the cell up to spanning K training samples.

> K is chosen as a function of N ($K_N$) and at infinite it tends to infinite but since is always smaller than the total N tending to infinte their ratio will be 0 $\lim_\infty\frac{K_N}{N}=0$

### Parzen Windows estimation

With a fixed volume V that can be introduced assuming temporarili that R is a n-dimensional hypercube.
if $h$ is the length  of an edge of that hypercube centered on x^* then its volume will be v=h^n

wSuccesively we create a window function with two results. If it gives 0 then our samole is outside otherwise it returns 1 when that x belongs to the hypercube. At the end to check if a training sample is part of our $R$ space we just apply to it our window function. fig3

THis particular estimator works as interpolator since it computes a gamma for every sample on the training set and it returns the global probability of the space sinnce it computes the window function for every sample belonging to our training set.
Fact is the hypercube can be quite limitating so we can compute an average of functions. This can be made by changing the window function in order to get a smoother density

#### WIndow width effect

We can observe that h controls the dimensions of the window and therefore the density

#### Properties

##### Bias

Since the samples x_i

fig 5

If we want to eliminate the blur we need a gamma function to be a dirac function. THis gives us an unbiased model but with the problem that it overfits the training sample

##### Variance

#### Gaussian Kernel

### Estimation with incomplete data

Until now we considered only estimation problems with complete data. Still there are cases with imcomplete observations.
Basically we repeat the process nade until now but repeated for every observation. The global parameters are ansed on a sum of every component.

In input we got $\theta$ which has all the parameters to be estiamted for every observation. Still since we don't have knowledge about the observation still we need to determine some parts.
This is due to the fact the observation are associated to all out $p(x)$