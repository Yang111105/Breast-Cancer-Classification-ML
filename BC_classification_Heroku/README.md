# BC_classification_Heroku
The purpose of this repository is to house the files necessary to process input from website users and return a prediction of Benign or Malignant. The research that is behind of this repo can be found [here](https://github.com/mpk9909g/Breast-Cancer-Classification). Please visit the link below to see our research page which is connected to our reserach GitHub:
(https://mpk9909g.github.io/Breast-Cancer-Classification/index.html) 

## Using this Repo on your local machine

1.Open git bash/terminal

1. Type ```source activate PythonData38```.
1. Hit ```enter```.
1. Type ```python app.py```.
1. Hit ```enter```.

See that the flask server is running.

Enter  http://127.0.0.1:5000/ in the chrome browser. 

Enter the required values to make prediction of whether a given population of breast tumor cells are malignant or benign. 

## Using the website that hosts this repo

Click this link : (https://bc-predict.herokuapp.com/). Enter the required parameters and press predict. 

This site uses *logistic regression* model as predictor with features selected based on correlation. 
The input features were optimized in a separate **GitHub Repo** that has a link to this website. For detailed information on feature selection and machine learning model optimization please click **[here](https://github.com/mpk9909g/Breast-Cancer-Classification)** and read the ReadMe. 
