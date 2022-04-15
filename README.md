# Breast-Cancer-Classification
This project uses machine learning algorithms to predict the class of a breast tumor based on specific features of a group of breast cells.

Click **[here](https://mpk9909g.github.io/Breast-Cancer-Classification/)** to go to the github page that hosts this repo and explore the website. That page has a link to our  Prediction site **http://bc-predict.herokuapp.com/** where you can predict the tumor type of a group of cells based on their features.
Click **[here](https://github.com/mpk9909g/BC_classification_Heroku.git)** to go to our second **GitHub-Repo** where you can find all necessary files and codes for our **Prediction Site**. 

## Notebooks in this Repo
Data preprocessing notebook includes data cleaning, feature selection and splitting the data into train and test sets.

Model comparison notebook evaluates all tuned Models against same test set and does **KFold Cross Validation** on each model with each feature group.

*Note: Although **Grid search** was done on Random Forest, while performing KFold Cross Validation the grid search tuning was not done due to size issues*. 

Notebooks for random forest model, KNN model,Logistic regression model,NN model and SVM model includes model optimization for each ML model. Each model has a confusion matrix that shows how well the model is doing in predicting the correct outcome. 

Every model is saved as .pkl files and stored in the ```pickles``` folder

## Models tested

**Logistic Regression** measures the relationship between the dependent variable and the independent variables, by estimating probabilities using its underlying logistic function.

**K nearest neighbors algorithm (KNN)** is an approach to data classification that estimates how likely a data point is to be a member of one group or the other depending on what group the data points nearest to it are in

**Random Forest**  is an ensemble method, meaning that a random forest model is made up of a large number of small decision trees, called estimators, which each produce their own predictions. The random forest model combines the predictions of the estimators to produce a more accurate prediction

**Support Vector Machine (SVM)** model is basically a representation of different classes in a hyperplane in multidimensional space. Given a set of training examples, each marked as belonging to one of two categories, an SVM training algorithm builds a model that assigns new examples to one category or the other, making it a non-probabilistic binary linear classifier (although methods such as Platt scaling exist to use SVM in a probabilistic classification setting). SVM maps training examples to points in space so as to maximise the width of the gap between the two categories. New examples are then mapped into that same space and predicted to belong to a category based on which side of the gap they fall.

**Neural Network** is a series of algorithms that endeavors to recognize underlying relationships in a set of data through a process that mimics the way the human brain operates. In this sense, neural networks refer to systems of neurons, either organic or artificial in nature. Neural networks can adapt to changing input; so the network generates the best possible result without needing to redesign the output criteria.

## Selected Feautures
We used 3 different groups as our selected features

First group includes all of the features from the data.
Second group includes the top 7 features selected based on the the SelectKBest function.
Third group includes the top 7 features selected based on their correlation with each other and the diagnosis.
We generated train and test data files and scaler files for all the feature groups.

All test and train datasets are in the test_train_data folder. All scaler files are in the scaler folder

Group 1; Train data files : X_train_scaled.csv Test data files : X_test_scaled.csv Scaler file scaler_allfeatures.sav

Group 2; Train data files : X_train_scaled_sk.csv Test data files : X_test_scaled_sk.csv Scaler file scaler_selectBestFeatures.sav

Group 3; Train data files : X_train_scaled_cs.csv Test data files : X_test_scaled_cs.csv Scaler file scaler_correlationFeatures.sav

## Rational of model optimization
We aimed for high accuracy with the least number of features.

Here you will see in addition to 3 groups of features used to train and test the various models also a KFold Cross Validation for further testing the models. 

We trained and tested our data with KNN, logistic regression, random forest, SVM and Neural network models. The winner is the logistic regression model where we used features selected based on their correlation with each other and with the diagnosis. This Model was then used in the Prediction site **http://bc-predict.herokuapp.com/**. 

## Architecture of the project

Please refer to the diagram to get an understanding of how each item is related in this project. It also shows how the two GitHub pages ( GitHub research and GitHub prediction) are connected.


![Diagram](/static/images/diagram.png)


## Reference:

1. https://www.kaggle.com/frankyd/machine-learning-with-breast-cancer
2. https://www.semanticscholar.org/paper/Multisurface-method-of-pattern-separation-for-to-Wolberg-Mangasarian/50537a16eb18e2bc4971165258cba7a071f38cb7
3. https://slideplayer.com/slide/15247250/
4. https://muditkapoor01.medium.com/classification-of-breast-cancer-82d0058f3d35
5. https://math.bu.edu/DYSYS/chaos-game/node6.html
6. https://www.analyticsvidhya.com/blog/2021/10/a-comprehensive-guide-on-deep-learning-optimizers/
7. https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
8. https://www.kenhub.com/en/library/anatomy/cell-nucleus
9. https://www.thoughtco.com/the-cell-nucleus-373362
10. http://www.captodayonline.com/Archives/feature_stories/1006benignmalignantRED.pdf
11. https://www.karger.com/Article/FullText/508780
12. https://deepai.org/machine-learning-glossary-and-terms/random-forest
13. https://en.wikipedia.org/wiki/Support-vector_machine
14. https://www.investopedia.com/terms/n/neuralnetwork.asp
15. https://machinelearningmastery.com/k-fold-cross-validation/
16. BioRender.com






