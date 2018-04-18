# PyCMF
Collective Matrix Factorization in Python.

Collective Matrix Factorization is a machine learning method that decomposes two matrices
 ![equation](http://latex.codecogs.com/gif.latex?X) and ![equation](http://latex.codecogs.com/gif.latex?Y) into three matrices
 ![equation](http://latex.codecogs.com/gif.latex?U), ![equation](http://latex.codecogs.com/gif.latex?V), and
 ![equation](http://latex.codecogs.com/gif.latex?Z)
 such that

![equation](http://latex.codecogs.com/gif.latex?X%20%5Capprox%20f%28UV%5ET%29)  

![equation](http://latex.codecogs.com/gif.latex?Y%20%5Capprox%20f%28VZ%5ET%29)

where ![equation](http://latex.codecogs.com/gif.latex?f) is either the identity or sigmoid function.

## Why Use CMF?
CMF decomposes complex and multiple relationships into a small number of components, and can provide valuable insights into your data. Relationships between
- words, documents, and sentiment
- people, movies, genres, and ratings
- items, categories, people, and sales

and many more can all be handled with this simple framework.
See [Use Cases](https://github.com/smn-ailab/PyCMF#use-cases) for more details.

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
$ pip install git+https://github.com/smn-ailab/PyCMF
```
Numpy and Cython must be installed in advance.

## Features
- Support for both dense and sparse matrices
- Support for linear and sigmoid transformations
- Non-negativity constraints on the components (useful in use cases like topic modeling)
- Stochastic estimation of the gradient and Hessian for the newton solver
- Visualizing topics and importances (see `CMF.print_topic_terms`)

See the docstrings for more details on how to configure CMF.

## Use Cases
See [samples](samples) for working examples.
Possible use cases include:

#### Topic modeling and text classification
Suppose you want to do topic modeling to explore the data, but want to use supervision signals such as toxicity, sentiment, etc.. By using CMF, you can extract topics that are relevant to classifying texts.

#### Movie rating prediction
Many prediction tasks involve relations between multiple entities. Movie rating prediction is a good example: common entities include users, movies, genres and actors. CMF can be used to model these relations and predict unobserved edges.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## References
[Lee, D., & Seung, H. (2001). Algorithms for non-negative matrix factorization.
Advances in Neural Information Processing Systems, (1), 556–562.](https://doi.org/10.1109/IJCNN.2008.4634046)

[Singh, A. P., & Gordon, G. J. (2008). Relational learning via collective matrix factorization.
Proceeding of the 14th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining
KDD 08, 650.](http://www.cs.cmu.edu/~ggordon/singh-gordon-kdd-factorization.pdf)

[Wang, Y., Yanchunzhangvueduau, E., & Zhou, B. (2017).
Semi-supervised collective matrix factorization for topic detection and document clustering.](http://ieeexplore.ieee.org/document/8005460/)

## TODO
- [ ] Improve performance
- [ ] Add support for weight matrices on relations
- [ ] Add support for predicting using obtained components
- [ ] Full compatibility with sklearn
