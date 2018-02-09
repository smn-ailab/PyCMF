# PyCMF
Collective Matrix Factorization in Python. 

At a high level, Collective Matrix Factorization is a machine learning method that decomposes two matrices
 ![equation](http://latex.codecogs.com/gif.latex?X) and ![equation](http://latex.codecogs.com/gif.latex?Y) into three matrices 
 ![equation](http://latex.codecogs.com/gif.latex?U), ![equation](http://latex.codecogs.com/gif.latex?V), and 
 ![equation](http://latex.codecogs.com/gif.latex?Z) 
 such that 
 
![equation](http://latex.codecogs.com/gif.latex?X%20%5Capprox%20f%28UV%5ET%29)  

![equation](http://latex.codecogs.com/gif.latex?Y%20%5Capprox%20f%28VZ%5ET%29)

where ![equation](http://latex.codecogs.com/gif.latex?f) is either the identity or sigmoid function.
Common uses include topic modeling, relation learning, and collaborative filtering. See [Use Cases](https://github.com/smn-ailab/PyCMF#use-cases) for more details.

## Usage
PyCMF implements a scikit-learn like interface (full compatibility with scikit-learn is currently in progress)

```python
>>> import numpy as np                                                                                          
>>> import pycmf 
>>> X = np.abs(np.random.randn(5, 4)); Y = np.abs(np.random.randn(4, 1))
>>> model = pycmf.CMF(n_components=4)
>>> U, V, Z = model.fit_transform(X, Y)
>>> np.linalg.norm(X - U @ V.T) / np.linalg.norm(X)
0.00010788067541423165
>>> np.linalg.norm(Y - V @ Z.T) / np.linalg.norm(Y)
1.2829730942643831e-05
```

## Getting Started
```bash
$ python setup.py install
```
Numpy, Scikit-learn, and Cython must be installed in advance.

## Features
- Support for both dense and sparse matrices
- Support for linear and sigmoid transformations
- Non-negativity constraints on the components (useful in use cases like topic modeling)
- Stochastic estimation of the gradient and Hessian for the newton solver
- Visualizing topics and importances (see `CMF.print_topic_terms`)

## Use Cases
Possible use cases include:

#### Topic modeling and text classification
By using CMF, we can extract topics that are important for classifying texts. (Tutorial coming soon)

#### Movie rating prediction
Many prediction tasks involve relations between multiple entities. Movie rating prediction is a good example: common entities include users, movies, genres and actors. CMF can be used to model these relations and predict unobserved edges.

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## References
[Relational Learning via Collective Matrix Factorization](http://www.cs.cmu.edu/~ggordon/singh-gordon-kdd-factorization.pdf)

[Semi-supervised collective matrix factorization for topic detection and document clustering](http://ieeexplore.ieee.org/document/8005460/)

## TODO
- [ ] Improve performance
- [ ] Add support for weight matrices on relations
- [ ] Add support for predicting using obtained components
- [ ] Full compatibility with sklearn
