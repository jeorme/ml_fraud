# Machile learning algorithms for fraud detection in credit card transactions

# Problem:
We are given a dataset containing the transaction made by credit cards in September 2013 by European cardholders. These contain 492 frauds out of 284,807 transactions.
We realise this data is highly unbalance: only 0.172% of all transactions account for frauds. This means that even a classifier that classifies everything as non-fraud would be 99.82% accurate! We need to be careful when analysing our results. We should aim to maximise the *recall*, this is how many of the true positives were found (recalled). Due to the nature of our problem, we'd rather have more false positives than miss frauds.

