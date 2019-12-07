## Classification-of-NBA-Rookiee-career-length

This project is a Classification problem of machine learning project. We explore the following Classification models to predict the target value and report the comparison of their performances based on F1 scores.

1. K-nearest neighbors
2. Random forests
3. Logistic regression
4. Artificial neural networks

### Data -
-------------------------------

NBA Rookie Stats dataset (provided at data.world) 
https://data.world/exercises/logistic-regression-exercise-1

Each row in the table represents a player’s rookie statistics for the first season. This dataset totals 21 columns and 1340 rows. 
The 21 features are play name (Name), games played (GP), minutes played (MIN), points per game (PPG), field goals made (FGM), 
field goal attempts (FGA), field goal percent (FG%), three points made (3PM), three point attempts (3PA), three point percent (3P%), 
free throws made (FTM), free throw attempts (FTA), free throw percent (FT%), offensive rebounds (OREB), defensive rebounds (DREB),
rebounds (REB), assists (AST), steals (STL), blocks (BLK), turnovers (TOV), and target (TAR).

Out of these 21 attributes, the last attribute is the y label (TAR) which needs to be predicted. It is a Boolean attribute, where “0” means the career 
length of the player is less than 5 years, and “1” greater than or equal to 5 years. 

### Program

The nba_ml.ipynb program can be executed in Jupiter notebook.

### Methodology

1. Importing the necessary packages
2. Loading the data and data inspection
3. Data visualization/ Addressing of the null value and Outliers
4. Inspection and addressal of correlation in features 
	Plotting correlation matrix
	Variation Inflation Factor (VIF)
	Principal Component Analysis (PCA)
5. Splitting of train/test dataset
6. Implementation of four types of classification models with GridSearchCV for finding the best hyperparameters
	K-nearest neighbors classifier
	Random forest classifier
	Logistic regression
	Artificial Neural Network
7. Performance comparison of various models

### Questions
1. During data preparation, did we discover any attribute to remove or any new attribute to add (feature engineering)? 

We found the presence of high multi-collinearity among the features. The multi-collinearity degrades the performance of the models. The multi-collinearity was checked by plotting collinearity matrix, computing and printing the VIF. A VIF value of 5 and above is considered to the presence of high collinearity. In our case we found the highest value of VIF to be 973 which is way higher than the collinearity definition. PCA is applied in the features to reduce the collinearity. The reduced the maximum VIF value to 1. The original features were twenty in number that reduced to nine after PCA. 

2. Normalizing (a.k.a., scaling) features is desirable for distance-based models, e.g., knearest neighbors. Did we try feature normalization for some of the models? If so, is there any improvement?

We performed Standard Scaling in features. Standard Scaling converts a feature to have zero mean and one standard deviation. We noticed a considerable increase in performance due to Standard Scaling. The F1 score of Logistic regression increased to 0.764 from 0.725 due to Standard Scaling. Also, the training time reduced by 10% due to feature scaling.

3. Regularization is a common practice to battle overfitting. How is varying the penalty parameter in logistic regression affect the performance F1 score on testing? (The logistic regression penalty parameter may be ’none’, ’l1’, ’l2’ or ’elasticnet’.)

Regularization greatly solves the issue of overfitting. L1 regularization is linear, L2 is quadratic, elasticnet is balance between L1 and L2 (li_ratio), and the none is no regularization.

We have run the logistic regression with all regularization techniques. The results are as follows - 

	. F1 score by Logistic Regression with l1 penalty is: 0.79
	. F1 score by Logistic Regression with l2 penalty  is: 0.8
	. F1 score by Logistic Regression with elasticnet penalty (l1_ratio = 0.5) is: 0.79
	. F1 score by Logistic Regression with none penalty is: 0.79

L2 regularization is giving a better F1 score for this example set by a marginal value.

4. These models have hyperparameters. When training, experiment using GridSearch to select hyperparameters for your models. What are the best hyperparameters among those we tried?

The best hyperparameters for the various models are -

	. KNN - best hyper parameter is: {'n_neighbors': 30}
	. Random Forest - best hyper parameters are: {'max_depth': 20, 'n_estimators': 100}
	. Logistic Regression -  best hyper parameters are: {'C': 1, 'penalty': 'l2', 'solver': 'saga'}
	. ANN - best hyper parameters are: {'alpha': 0.001, 'hidden_layer_sizes': (3,), 'learning_rate_init': 0.001}

5. Which model gave the best F1 score on testing data?

. Best F1 score by KNN is: 0.68
. Best F1 score by Random Forest is: 0.75
. Best F1 score by Logistic Regression is: 0.8
. Best F1 score by ANN is: 0.78

The model with the best F1 score and the corresponding score is: ('Logistic Regression', 0.8)
