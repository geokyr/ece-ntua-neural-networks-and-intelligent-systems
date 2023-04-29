# ntua-neural-networks

Lab Assignments for the [**Neural Networks and Intelligent Systems**](https://www.ece.ntua.gr/en/undergraduate/courses/3319) course, during the 9th semester of the **School of Electrical and Computer Engineering at the National Technical University of Athens**.

## Team 44 - Members

- [**Kyriakopoulos George**](https://github.com/geokyr)
- [**Tzelepis Serafeim**](https://github.com/sertze)

## Lab 01 - Supervised Learning: Classification

The first lab was about studying and optimizing classifiers on two different datasets, the [HCC Survival](https://archive.ics.uci.edu/ml/datasets/HCC+Survival) and the [kdd cyberattack](https://www.kaggle.com/datasets/slashtea/kdd-cyberattack) dataset.

The optimization of the first dataset was done using exclusively the [scikit-learn](https://scikit-learn.org/stable/) library, while the second one was done using the [Optuna](https://optuna.org/) library.

The classifiers used were:

- Dummy
- Gaussian Naive Bayes (GNB)
- KNearestNeighbors (KNN)
- Logistic Regression (LR)
- Multi-Layer Perceptron (MLP)
- Support Vector Machines (SVM)

while for metrics it was mainly accuracy and F1 score.

## Lab 02 - Unsupervised Learning: Content-Based Recommender System and SOM-Based Data Visualization

The second lab was about implementing a content-based recommender system for movies and creating a SOM for data visualization. The [Carnegie Mellon Movie Summary Corpus](http://www.cs.cmu.edu/~ark/personas/) dataset was used for this purpose. 

We implemented recommenders based on the TF-IDF and Word2Vec algorithms, utilizing transfer learning for the embeddings from the [Gensim](https://radimrehurek.com/gensim/models/word2vec.html) library, for the latter. To train the SOM, we used the [Somoclu](https://somoclu.readthedocs.io/en/stable/index.html/) library.

## Lab 03 - Deep Learning: Image Captioning

The third lab was about implementing and optimizing an image captioning system. The [Flickr30k](https://www.kaggle.com/hsankesara/flickr-image-dataset) dataset was used for this purpose.

The image captioning system was based on a transformer-based model and BLEU score was used as the metric for optimization.

The optimizations included:

- trying different encoders
- modifying the preprocessing of the captions
- utilizing transfer learning for the embeddings from the [Gensim](https://radimrehurek.com/gensim/models/word2vec.html) library
- using a beam search algorithm for sentence generation
- trying different hyperparameters for the model
