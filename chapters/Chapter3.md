# Chapter 3 - Feature Reduction

## Introduction

(img graph slide 11)

## Feature Reduction

In order to reduce the dimensionaluty, as said before, we need to reduce the number of features found. The aim while doing this is to:

> *"Minimize the number of features while keeping the discrimination capability as higher as possible"*

To achieve this there are two main approaches we can take:

- Feature reduction by selection (*Feature selection*)
- Feature reduction by transformation (*Feature reduction*)

### Problem Formulation

Let $\bold{F} = \{x_1,.........x_m\}$ be the set of $n$ available features where $f_k$ and $f_{\underline{k}}$ the $k$-th selected and discarded features from $F$,

The goal is to select a $subset \space \bold{F^*}$ composed of $m<n$ features such that 

$$ \bold{F^*} =\underset{\bold{F'}\in \bold{F}, \space card(\bold{F'}) = m}{arg \space max} \{ J(\bold{F'}) \}$$