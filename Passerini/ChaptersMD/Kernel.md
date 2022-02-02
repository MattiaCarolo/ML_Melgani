---
title: "Kernel - Passerini"
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

# Kernel Machines

For learning non-linear models we can apply feature mapping (you have to know _which_ mapping to apply). Even if polynomial mapping is useful, in general it could be _expensive_ to explicitly compute the mapping and deal with a high dimension feature space.

If we look at dual formulation of SVM problem, <u>the feature mapping only appears in dot products</u>.
The **kernel trick** replace the dot product with an equivalent kernel function over the inputs, that produces the output of the dot product but in feature space, without mapping (explicitly) $\bold  x$ and $\bold x’$ to it.