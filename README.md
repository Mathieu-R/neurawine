# neurawine

The `wine dataset` from `sklearn` is a result of a chemical analysis of different wines grown in the same region of Italy by `3 different cultivators`. Each sample has `13 measurements` (_alcohol, magnesium, color intensity,...). The target value of each sample is one of the three cultivators. 

The goal of this neural network is to predict from which of the 3 cultivators a given wine comes based on these 13 measurements.

### Installation

Clone the project

```bash
$ git clone https://github.com/Mathieu-R/neurawine
```

Create virtual environment

```bash
$ python3 -m venv <env-name>
$ source env/bin/activate
$ python3 -m pip install --upgrade pip
```

Install required packages

```bash
$ python3 -m pip install -r requirements.txt
```

### Interesting reading

- http://neuralnetworksanddeeplearning.com/chap2.html
- https://peterroelants.github.io/posts/neural-network-implementation-part04/
- https://python-course.eu/machine-learning/neural-networks-structure-weights-and-matrices.php
- https://ml-cheatsheet.readthedocs.io/en/latest/
- https://machinelearningmastery.com/how-to-improve-neural-network-stability-and-modeling-performance-with-data-scaling/
