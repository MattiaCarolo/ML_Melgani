# Cheat Sheet

## Gradient Descent
Made in two steps:

(1) Initialize $\bold w$ 

(2) iterate until gradient is approx. zero
$$
\bold{w} = \bold{w} - \eta\nabla E(\bold{w};D)
$$

## Rules on gradient

Simple application
$$
\nabla_a a^T a = 2a
$$

Applied to MSE
$$
\nabla (\mathbf{y}-X\mathbf{w})^T(\mathbf{y}-X\mathbf{w}) = 2(y - X\mathbf{w})^T(-X)
$$

here the gradient will be 2 times the transpose times the gradient of the second term which in this case since we are doing it respect to $w$ we get $X$. The choice of applying the gradient to the transposed term or the other is base on whether we want to a row or column gradient.

## Algebra
$$
(xw)^T = x^T w^T
$$
