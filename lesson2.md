# Lezione 2 Melgani

## ML System Supervised

In every ML system data comes from the physical world when I'm acquiring information (we do not consider simulation).

During the **acquisition** we have a wide range of sensors where the choice of the singular sensor it's based on waht we need to measure. Every sensor has different feature that can make one better than other. Key features are:

- Bandwith
- Sensibility
- Noise

After that there is a **pre-processing** part where we manage the data in order to "clean" the unwanted part of the signal. If a sensor has an extreme quality we could skip the pre processing part. During this part we can manipulate a lot of data until we have a clean signal.

At the end of the preprocessing we can extract the features from our signal in order to find the inner data. During this part called **feature extraction** we need to understand which feature to extract in the signal.
e.g. In a audiowave we can use as feature the amplitude of the wave like anything else as long it satisfies our needs.
Features must be insensitive to scale or rotation or manipulation that do not change the inner data of the signal but only the reference system.
We can't take too many features so we go during a subphase called **feature reduction** which cuts the number of features.

With a subset of features we arrive at the object space defined by the user. Here we classify the object with his features. The choice of the classifier depends on many aspects:

- Number of features
- How many classes
- Can classes overlap

Than after generating an output where we classified the data we can check the quality of the process during **post-processing** where we can "correct" some bad classification.

At the end we have the **decision** where it is submissed to hte supervisor in order to check if the outcome is correct. If not there will be changes in one or more of the steps described before. To do this the supervisor can change features, sensors, different classificiation and so on.

Usually the basic system it's implemented starting with a simple base in order to make already in early stages the right adjustments

## Design Phases

Designin' an ml system it's composed by vary phases but all have in common one thing. To design something it's IMPORTANT to have prior knowledge about the problem

### 1) Collecting Data

By this we mean collecting training samples. It's needed to collect the ground truth where we know that for a given $x$ taken with the signals we have an $y$ which represents the ground truth in order to get the couple $(x,y)$ where we know the exact result for a given input.
During this phase we need to have relevant samples for our system so always during this part we take only the desired part. All the sequent parts are dependant from this phase so it's important to collect the right data.

### 2) Select features

self explanatory

### 3) Choose Model

We select the model to apply. Usually it starts with a linear model and it increases it's complexity with the continuos interation and features to consider

### 4) Train classifier

The classifier it's a mathematical model and during this phase we need to evalaute the parameters.

### 5) Evaluation

Even here pretty self explanatory. It can be done in terms of accuracy or in terms of processing time. This choice it's heuristically made based on the circumstances.


During the training phase the classifier should estimate the best linear separation according to the adopted cost function. Add to this that we need to give a weight to every error in order to create a cost function so we have a value that gives us an idea of how much our model fits in the ps