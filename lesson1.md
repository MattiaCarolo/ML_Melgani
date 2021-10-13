# Machine learning mod.1

 Melgani,passerini,ricci

## Course module

- Introduction
- Density estimation
- Feature Reduction
- Classification
- Linear Discrimannt analysis

(Melgani)

- Kernel Machines
- Learning Bayesan Network
- Reinforcement Learning
- Unupervised learning

(Passerini)

The exam can be done after the end of the first module asking for an appointment

NB in the second module there will be a group project (2-3 persons)

If u fail a part u can take it again but on rejection u need to repeat all the parts (kek)

## Signals

A signal is a conveyor of inforamtions coming from a process (it doesn't matter what kind of process) of our interest

### monodimensional signal

function $f(\epsilon)$ represnting the evolution of the information correlated to the variable $\epsilon$ which translates a physical reality such as time, dimensions etc.etc.

### multidimensional signal

There are multiple correlated or uncorrelated physical realities -> $g(\epsilon,\theta,\alpha ....)$

An image can be a signal where the information can be the coordinates like $x,y$ and so on so forth

### Signal Acquisition

To acquire physical signals from a process we need a sensor who converts the physical signal to a digital signal. Every sensor has different qualities and properties like the sensivity which measures the minimum level of signal in order to catch it.

Since a signal is a function we have 2 approaches:

- Continuous polling of our process
- Digitalize our signla

In order to digitalize we take a finite numbers of sample. By doing this there is a loss information but it cant be mitigated (not part of our course). After that we need to discretize across the magnitude dividing it by levels. With dis we are quantizing the information. In poor words after we take a sample we look at what level of our magnitude the sample is more near and after that we assign the sample our level set before.

During sampling and quantization we lose some information due to our finite states.
In the end after all the process we can put the signal in the memory assigning to our levels a codification in binary so when the sapmple is recognized during the quantization it receives its codification in bit

### Example of 1d signal

Waveform of the heartbeat

### Example of 2d signal

Every image can be a 2d signal. In this case we can subdivide out image in N parts where N is the number of sensors at our disposal. After that all info in our subimage is convoluted to our sensor in order to activate it.
As common people look to take information from the vision domain during an analyze of an image we take information from all domain of our interest like UV or microwaves.

During the analysis there is no obligation to convolute the channels in order, so we can emphasize features combining different channels of our sensors.

### Example of n-d signal

We can create the same image with different bands in order to extract different features. In order to do this we can utilize a great amount of filters

## Learning

"Machine learning (ML) is the study of computer algorithms that can improve automatically through experience and by the use of data. It is seen as a part of artificial intelligence. Machine learning algorithms build a model based on sample data, known as "training data", in order to make predictions or decisions without being explicitly programmed to do so" Wikipedia

Our learning is subdivided in 5 main categories

### Supervised Learning

The learner is fed with a set of input/output pairs (training set)
If input is our $x$ and ouput $y$ we need to give every $x$ value an estimate of our $y$ value

#### input space

Suppose you take an image of a face. The first thing to do is extract a set of features in order to be able to classify the face we have. This feature can be mouth width, eyes colors etc.etc creating our input space or feature space (very general take care because features can vary after input)

#### output space

If we need to decide for example if someone can enter a building the output space is the union of all possible outcomes we decided

### Unsupervised Learning

No output information is provided, but just input data which are modeled by the learner (data clustering). We can use different type of correlation in order to distinguish cluster starting from similarity methods(Louvain) to dissimilarity methods

### Semi-supervised Learning

Even if there is a limited pool of input/ouput data we have a batch of unlaballed data which needs to be classified. We use our training samples to train our learner and then we apply it to our unlabelled batch. The labelling is done during the learning process

### Active Learning

In this case the base is the same of the supervised learning with a twist. During the learning the learner interacts with a supervisor named oracle whom checks and labels the results found by our learner.

### Reinforcement Learning

This is a scenario of software agents. These agents can do actions in a closed environment where every action gives a reward. The agent seeks to maximize the rewards gived by the environment. An agent performs a set of different actions in order to understand how the environment responds to different actions to maximize the global reward of our agent

## Supervised Task

### Binary classification

Assume an image where there could be only two possible outcomes of the result (recognize ground from building).

### Multiclass classification

Non ho voglia fa a casa 

### Multilabel classification

It takes single parts of our image and then it confronts that with a binary vector based on the labels stored. For each label it signals if that label can be associated or not

### Captioning

Commenting the image based on the features found

### Regression

Used to estimate a scalar value from a continuos input. 

### Ranking

Used to rank observations. It responds to the question "who is more likable to meet certain requirements"
