# Fraud detection - machile learning algorithms with a quantum twist
## Task:
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
Another issue that arises from this highly unbalanced data is poor training: there are too few examples of the minority class for a model to effectively learn the decision boundary. To fix this we explored [SMOTE](https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/) (Synthetic Minority Oversampling TEchnique), which is a widely used approach to synthesizing new examples of the minority class (frauds).

### Results (preliminary):
We were only able to obtain results for the classical machine learning and neural network algorithms with no SMOTE.
<p align="center">
<br/><img src='/ml_clas_results.png' width="450">
</p>
<p>
<em>Results for the classical ML algorithms with no SMOTE. The K-NN algorithm proves to be the best performing in terms of accuracy, F1 and recall. However, due to the unbalance data issue, only 80 fraud cases were present in our test data, so the results need to be taken with a pinch of salt.</em>
</p>
The results obtained to the classical neural network were in a similar success range to those of the classical regressive methods used in the above picture. I believe that for this relatively simple classification problem a neural network may not be necessary (I would also need more time and understanding to choose the optimal batch size/epoch/optimizer/activation combination).
When incorporating SMOTE we ran into convergence issues and large computational times, which made it impossible to obtain results within the timeframe allocated to the project. We were only able to obtain SMOTE results for K-NN, whose recall improved from 82.50% to 87.5%.

The preliminary results of the hybrid algorithm suggested it was working correctly, but it had to be run on only 10% of the data set due to large computational times.
<p align="center">
<br/><img src='/prel_hybrid.jfif' width="450">
</p>
<p>
<em>Preliminary training results of the hybrid machine learning code for only 10% of the total data set, no SMOTE. The algorithm seems to be working correctly but computational times are too large.</em>
</p>

### Requirements:
Our script was run on Conda 4.10.01
### Problems:
- Our neural network algorithms (both cassical and hybrid) are too time expensive to be run with the full dataset.
- SMOTE creates additional data (approximately double) which causes larger computational times for convergence, but also outperforms non-SMOTE training.
- The PennyLane layer activation recommendations caused very low accuracy.

### Future work and notes:
- Changing the amount of oversampling could have reduced convergence issues when using SMOTE.
- Try different combinations of hybrid NN.
- Explore different optimization and activation parameters in the NN.


