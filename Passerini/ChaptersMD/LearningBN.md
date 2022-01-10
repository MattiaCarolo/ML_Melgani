---
title: "Learning in Graphical Models - Passerini"
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

# Parameter estimation

Let's assume that we are given a model with already his structure. Together we are given also a dataset of examples $D=\{x(1),...,x(N)\}$ where each example $x(i)$ is a configuration for *all* (complete data) or *some* (incomplete data) variables in the model.

Now we need to estimate the parameters of the model from the data given. The simplest approach is to do Maximum Likelihood Estimation of the data given so
$$
\theta
$$