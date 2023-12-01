# Soldiers Race Prediction
<img src="https://github.com/mbarnawi/ML_Portfolio/blob/main/Soldiers%20Race%20Prediction/images/Capture.PNG" width=600>
This machine learning project focuses on imbalanced multiclass classification using supervised learning techniques. The objective is to predict the race of a soldier based on a set of given features.

# Used Libraries

- [NumPy](https://numpy.org/) - [Pandas](https://pandas.pydata.org/) - [Matplotlib](https://matplotlib.org/) - [Seaborn](https://seaborn.pydata.org/)
- [scikit-learn](https://scikit-learn.org/) - [scikit-plot](https://scikit-plot.readthedocs.io/) - [imbalanced-learn](https://imbalanced-learn.readthedocs.io/)
- etc
# Run the project
To run the Jupyter Notebook in the repository on your local machine or using Google Colab

# Results 
<img src="https://github.com/mbarnawi/ML_Portfolio/blob/main/Soldiers%20Race%20Prediction/images/both.PNG" width=1000 highet=200>


The tuned SVM model achieved excellent scores and performed better than three other tuned models (Logistic Regression, Random Forest, XGBoost), particularly in terms of the minority class
 The image provided displays the Recall-Precision Curve between the top two performing models, demonstrating that the tuned logistic regression model exhibits greater consistency.

 ## Optimal Hyperparameters for Logistic Regression:
 | Parameter      | Value       |
| -------------- | ----------- |
| C              | 1.5           |
| class_weight   | 'balanced'  |
| gamma      |  'scale'    |
| kernel         | 'linear'       |

## Comparing Hispanic Scores for SVM and Logistic Regression:
| Model                | F1_Hispanic | Precision_Hispanic | Recall_Hispanic |
| -------------------- | ----------- | ----------------- | --------------- |
| Logistic Regression  | 0.673    | 0.722          | 0.691        |
| SVM                  | 0.569    | 0.722         | 0.725        |


# Acknowledgements
This project is part of ML bootcamp provided by [clarusway](https://clarusway.com/)





