import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from pickle import dump, load


breast_cancer_df = pd. read_csv("breast_cancer_data.csv")

# Drop the null columns where all values are null
breast_cancer_df = breast_cancer_df.dropna(axis='columns', how='all')
# Drop the null rows
breast_cancer_df = breast_cancer_df.dropna()

# define the output as target
target = breast_cancer_df["diagnosis"]

# features = breast_cancer_df[['radius_mean', 'texture_mean', 'perimeter_mean',
#     'area_mean', 'smoothness_mean', 'compactness_mean', 
#     # 'concavity_mean',
#     #    'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
#     #    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
#     #    'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
#     #    'fractal_dimension_se', 'radius_worst', 'texture_worst',
#     #    'perimeter_worst', 'area_worst', 'smoothness_worst',
#     #    'compactness_worst', 'concavity_worst', 'concave points_worst',
#     #    'symmetry_worst', 'fractal_dimension_worst'
#     ]]

features = breast_cancer_df[['perimeter_worst',
'concave points_worst',
'symmetry_worst',
'smoothness_worst',
'compactness_worst',
'texture_worst',
'fractal_dimension_worst'
       ]]

# split the data
X_train_cs, X_test_cs, y_train_cs, y_test_cs = train_test_split(features, target, random_state=42)

# scale the data
X_scaler_cs = StandardScaler().fit(X_train_cs)

# Transform the training and testing data using the X_scaler model
X_train_scaled_cs = X_scaler_cs.transform(X_train_cs)
X_test_scaled_cs = X_scaler_cs.transform(X_test_cs)

# Label-encode data set
label_encoder = LabelEncoder()
label_encoder.fit(y_train_cs)
encoded_y_train_cs = label_encoder.transform(y_train_cs)
encoded_y_test_cs = label_encoder.transform(y_test_cs)


# test with logistic regression
logistic_model = LogisticRegression(random_state = 0)
logistic_model= logistic_model.fit(X_train_scaled_cs, encoded_y_train_cs)

# Save the model to a pickle file (i.e., "pickle it")
# so we can use it from the Flask server.
dump(logistic_model, open('logistic_model.pkl', 'wb'))
 


# Save the scaling function to a pickle file (i.e., "pickle it")
# so we can use it from the Flask server. 
dump(X_scaler_cs, open('scaler_cs.pkl', 'wb'))




# # Load the scaler.
# scaler = load(open('scaler_cs.pkl', 'rb'))