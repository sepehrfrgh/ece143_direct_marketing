# random forest to rate feature importance
    1. The principle of feature importance obtained by random forest
The calculation method of the importance of a feature X in a random forest is as follows:

1: For each decision tree in the random forest, use the corresponding OOB (out-of-bag data) data to calculate its out-of-bag data error and record it as errOOB1.

2: Randomly add noise interference to the feature X of all samples of the OOB data outside the bag (you can randomly change the value of the sample at the feature X), calculate the error of the data outside the bag again, and record it as errOOB2.

3: Suppose there are Ntree trees in the random forest, then the importance for feature X = ∑ (errOOB2-errOOB1) / Ntree. The reason why this expression can be used as a measure of the importance of the corresponding feature is because: After a feature is randomly added with noise, the accuracy outside the bag is greatly reduced, which indicates that this feature has a great impact on the classification result of the sample, that is, its importance is relatively high.