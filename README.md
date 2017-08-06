# toadstool
Predicts whether a mushroom is edible or toxic using classification models

## Plotting
Using the `learn` module we can visualize the performance of different classification models.

For example, we can use `train_percent_accuracy` to plot the relationship between each model's
dataset training percentage and its accuracy score.

![train_percent_accuracy.png](images/train_percent.png "Training percentage vs. accuracy score")

The lower accuracy at lower training percentages can be explained by not utilizing enough training data
to effectively train each model. 

As can be seen, Random Forest and Decision Tree classification were the most accurate, reaching
scores of 1.00.

We can use `feature_importances` to visualize the importance of each feature when trained with
the Random Forest Classifier:

![feature_importance.png](images/feature_importance.png "Features vs. Importance in Random Forest")

The performance of each model can also be quantified. `MultiClassifier.train_all` populates the dictionary
`MultiClassifier.performances` with the total time taken to train each model, which can then be plotted
using `performance_all`:

![performances_all.png](images/performances_all.png "Model vs. Performance (s)")