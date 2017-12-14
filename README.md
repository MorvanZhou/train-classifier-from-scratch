# Train a car condition classifier from scratch

This is a step by step tutorial of training a neural network classifier.
The dataset I'm using is from [UCI](http://archive.ics.uci.edu/ml/datasets/Car+Evaluation).
I've also wrote a detailed tutorial in Chinese in [莫烦Python](https://morvanzhou.github.io/tutorials/machine-learning/ML-practice/build-car-classifier-from-scratch1/).

## Data description

**4 Classes about car's condition:**
* unacc: unaccepted condition
* acc:  accepted condition
* good: good condition
* vgood: very good condition

**Features:**
* buying: vhigh, high, med, low.
* maint: vhigh, high, med, low.
* doors: 2, 3, 4, 5more.
* persons: 2, 4, more.
* lug_boot: small, med, big.
* safety: low, med, high.

## Training
**Files:**
* [data_processing.py](/data_processing.py) : download data and process to an accepted format
* [model.py](/model.py) : training model and view result

![Training result](/result.png)


## Dependencies
* Python
* tensorflow
* pandas
* numpy
* matplotlib

You can view more tutorials on [this page](https://morvanzhou.github.io/) or know more about me on [here](https://morvanzhou.github.io/about/).