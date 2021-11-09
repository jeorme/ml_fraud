# Fraud detection - machile learning algorithms with a quantum twist
## Problem:
Our goal is to exploit the power of machine learning for fraud detection. In particular, we want to explore the potential of quantum computing for tackling this classical problem.<\br>
For this project we will use a dataset containing the transaction made by credit cards in September 2013 by European cardholders. More information on the data set can be found [here](https://www.kaggle.com/mlg-ulb/creditcardfraud).
### Approach:
We explored 3 different approaches to solving this problem:
- Classical:
   * **Regression models**: decision tree, KNN Logistic Regression, SVM, Random forest tree, XGBoost. The implementation of this models can be found in *classical_ml.py*.
   * **Neural networks**: purely classical neural network with 3 layers. Found in *classical_nn.py*.
- Hybrid (classical + quantum):
   * **Hybrid neural network**: we incorporate a quantum layer in between 2 classical layers in our neural network. This is done in *hybrid_nn.py*.
   This is done by converting a PennyLane quantum layer into a Keras layer. We use the KerasLayer class of the qnn module, which converts the QNode to the elementary building block of Keras: a layer. For more information on this procedure visit the PennyLane tutorial [here](https://pennylane.ai/qml/demos/tutorial_qnn_module_tf.html).

### Sampling issue:
These contain 492 frauds out of 284,807 transactions. We realise this data is highly **unbalanced**: only 0.172% of all transactions account for frauds. This means that even a classifier that classifies everything as non-fraud would be 99.82% accurate! Therefore, we need to be careful when analysing our results. We should aim to maximise the *recall*, which represents how many of the true positives were found (recalled). Due to the nature of our problem, we'd rather have more false positives than missed frauds.<\br>
Another issue that arises from this highly unbalanced data is poor training: there are too few examples of the minority class for a model to effectively learn the decision boundary. To fix this we explored [SMOTE](https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/) (Synthetic Minority Oversampling TEchnique), which is a widely used approach to synthesizing new examples of the minority class (frauds). SMOTE proved to be a very successful aproach to solving our low minority sampling issue, as it increased accuracy and F1 score (INCLUDE DATA).


