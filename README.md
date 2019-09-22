# Santander Customer Transaction Prediction

## Project Background
Santander Bank is continually challenging machine learning algorithms to identify new ways to solve our most common challenge, binary classification problems such as: is a customer satisfied? Will a customer buy this product? Can a customer pay this loan? <br>

Santander published this task on Kaggle and invite Kagglers to identify which customers will make a specific transaction in the future, irrespective of the amount of money transacted. The data provided for this competition is anonymized but has the same structure as the real data which is available to solve this problem.

## Reuiqred Libraries
- numpy, pandas: numeric data and dataframe operation
- scipy: statistic 
- matplotlib, seaborn: data visualizatin
- sklearn: sci-kit Learn for machine learning
- lightgbm: lightgbm algorithm
- bayes_opt: bayesian optimization

## Data Exploration Analysis
- Define numerical and categorical variables： All the features are numerical.
- Target distribution
- Missing values
- Unqiue values
- Feature correlation
- Normality check (*scipy.stats.normaltest(), sns.distplot*):  All 200 features in train and test datasets are normally distributed.

## Feature Engineering
Unfortunately most of the feature engineering techiniques didn't seem to work in this task for tree-based models. However, feature scallings are still quite effective for linear models

### Standardization 
Both time and cross validation score (which use logistic regression and roc-auc scoring) has been improved with feature standardization

### Feature impportance 
- From logistic regression *(LogisticRegression(solver='lbfgs').fit()*
importance from coefficient
- From lightgbm *(lgb.LGBMClassifier.fit())*
importance按照某一个特征在构建决策树的过程中参与分裂的次数来决定的，分裂次数越多越重要<br>
综合来看， var_81,139,6,12,53,174,166,76,34 是比较重要的几个feature.

## Train the model - LightGBM
evaluated by roc_auc_score, auc = 0.897788089.

## Model Tuning
### Manual tuning and retrain
best parameters: 
- 'learning_rate': 0.01, 
- 'metric': 'auc', 
- 'seed': 42, 
- 'objective': 'binary',
- 'boost_from_average':'false',

- 'feature_fraction': 0.1, 
- 'bagging_freq': 1, 
- 'num_leaves': 2, 
- 'max_depth': 3, 
- 'min_gain_to_split': 0, 
- 'bagging_fraction': 0.4, 
- 'min_sum_hessian_in_leaf': 30, 
- 'lambda_l2': 0.01, 
- 'lambda_l1': 0.01
#### Evaluation
After retrain with the manual tuned parameteters, we got a training's auc= 0.913497 and	valid_1's auc= 0.898681. We predicted on the validation dataset and evalutated with validation targets using 'roc-auc_score' and got a mean validation auc = 0.900402356. <br>

*roc_auc_score(  y_valid, p_valid = lgb_clf.predict(X_valid)       )* <br>

### Auto tuning - BayesianOptimization
best parameters: <br>
- 'bagging_fraction': 0.6919921695505935,
- 'feature_fraction': 0.0893374972633,
- 'lambda_l1': 0.2176903455569933,
- 'lambda_l2': 0.615088537406593,
- 'max_depth': -1.0,
- 'min_gain_to_split': 0.9549795470898514,
- 'min_sum_hessian_in_leaf': 97.87877733741233,
- 'num_leaves': 2.9498467435595215
#### Evaluation
After Auto tuning, we got a training's auc= 0.914706 and	valid_1's auc= 0.898381, which is very similar and only a little lower than manual tuned reslut.

## Stacking
1) Create 3 LightGBM models with top 3 tuned parameters and create out-of-fold predictions for the above 6 models, as Level 1 model. We got auc = 0.899724

2) Train a Logistic Regression as level 2 model with level 1 features only. we got an auc = 0.8996087382850039

3) Train a LightGBM as level 2 model with level 1 features and raw features, and got 






