# Breast Cancer Classification: Project Overview 
* Created a machine learning model that classifies whether breast cancer is Malignant or Benign.
* Analyzed dataset with 569 observations and 33 columns
* Processed the data for making it suitable of better machine learning
* Optimized Logistic, and Random Forest Regressors using RandomSearchCV and cross validation to reach the best model. 
* Made a sample prediction for a given observation

## Code and Resources Used 
**Python Version:** 3.10.7  
**Packages:** pandas, numpy, sklearn, matplotlib, seaborn 

## Dataset Information
Dataset contains 32 features that are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image.

Link to dataset - https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data

## EDA
Used count plot for finding out whether the given dataset is balanced. I looked at the distributions of the features and analyzed correlation of the variables for finding out multicollinearity.
 Below are a few highlights from exploratory data analysis. 

**Count Plot of Output**  

![alt text](https://github.com/mubarakmayyeri/breast-cancer-classification/blob/master/images/cancer_count.jpg "No. of observations for each output")  
**Correlation Heatmap**

![alt text](https://github.com/mubarakmayyeri/breast-cancer-classification/blob/master/images/correlation.jpg "Correlation Heatmap")


## Data Cleaning
After collecting the data, I needed to clean it up so that it was usable for our model. I made the following changes to the variables:

*	Removed unnecessary columns
*	Encoded categorical variables
*	Removed features with multicollinearity 
*	Extracted features and target 
*	Standardized the dataset


## Model Building 

First I split the data into train and tests sets with a test size of 20%.   

I tried two different models and evaluated them using Accuracy score. Fine tuned model with best accuracy for better machine learning process. 

Machine Learning models:
*	**Logistic Regression** – chose this model because our problem is binary classification
*	**Random Forest** –  with default parameters

## Model performance
The Logistic regression model outperformed the Random forest model on the test and validation sets. 
*	**Random Forest** : Accuracy = 94.73 %
*	**Linear Regression**: Accuracy = 92.98 %

Chose **Logistic regression as final model** this problem and fine tuned its hyperparameters.
*   **Tuned Logistic regression** : Accuracy = 94.73 %, Precision = 91.48 %, Recall = 95.55 %, F1 Score = 93.47 %

![alt text](https://github.com/mubarakmayyeri/breast-cancer-classification/blob/master/images/scores.jpg "Performance Metrics of Models")

Final model gave same accuracy as the model with default parameters, while precision score was little lower the recall and f1 score improved.
The model made 108 correct predictions and 6 wrong predictions on the test data.

## Sample Prediction
I made a sample prediction with the Tuned Logistic regression model with the given features:  

radius_mean = 11.13, texture_mean = 16.62, smoothness_mean = 0.08151, compactness_mean = 0.03834, symmetry_mean = 0.1511, fractal_dimension_mean = 0.06148, radius_se = 0.1415, texture_se = 0.9671, smoothness_se = 0.005883, compactness_se = 0.006263, concave points_se = 0.006189, symmetry_se = 0.02009, symmetry_worst = 0.2383

![alt text](https://github.com/mubarakmayyeri/breast-cancer-classification/blob/master/images/sample_prediction.jpg "Sample prediction")

The machine learning model predicted that the cancer with given observations is **Benign**