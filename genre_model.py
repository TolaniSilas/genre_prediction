# Import the necesaary libraries and modules.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb  # Import XGBoost
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score, roc_auc_score, f1_score, recall_score, make_scorer, auc, roc_curve, precision_recall_curve, average_precision_score
from sklearn.preprocessing import LabelEncoder


# Import the csv file.
echo_tracks = pd.read_csv("genre_dataset.csv")

# Create features(predictor) variables.
features = echo_tracks.drop(["Unnamed: 0", "genre", "track_id"], axis=1)

# Create labels(outcome).
labels = echo_tracks.genre

#Initialize LabelEncoder.
label_encoder = LabelEncoder()   

# Fit and transform the labels.
encoded_labels = label_encoder.fit_transform(labels)  # 0 -- Hip Hop | 1 -- Rock

# Columns of feature variables.
columns = features.columns

# Split our data into training and testing sets.
train_features, test_features, train_labels, test_labels = train_test_split(features, encoded_labels, random_state=20202024, test_size=0.2, shuffle=True, stratify=encoded_labels)


# Instantiate the StandardScaler class.
scaler = StandardScaler()

# Scale train_features and test_features.
scaled_train_features = scaler.fit_transform(train_features)
scaled_test_features = scaler.transform(test_features)

# Convert the numpy sacaled data into a dataframe.
scaled_train_features = pd.DataFrame(scaled_train_features, columns=columns)
scaled_test_features = pd.DataFrame(scaled_test_features, columns=columns)

# Define the generated best model parameters.
xgb_params = {
    'subsample': 0.7,
    'reg_lambda': 0.8,
    'reg_alpha': 0.5,
    'n_estimators': 490,
    'min_child_weight': 1,
    'max_depth': 3,
    'learning_rate': 0.06918367346938775,
    'gamma': 0.11,
    'colsample_bytree': 0.7
}

# Instantiate the XGBoost Classifier with best parameters.
xgb_model = xgb.XGBClassifier(**xgb_params, booster="gbtree", eval_metric=['logloss', 'error'], objective="binary:logistic", random_state=20202024)

# Fit the training sets.
xgb_model.fit(scaled_train_features, train_labels)


# Define the function to perform the prediction.
def predict_class_proba(user_input):
    "A function that predict the probabilities of each classes(positive(Rock) and negative(Hip Hop)) based on the collected input features."

    user_input = np.array([user_input])
    user_input = user_input.reshape(1, 8)

    # Scale the user input using the same scaler used during training.
    scaled_input = scaler.transform(user_input)

    # Make Prediction probability.
    class_proba = xgb_model.predict_proba(scaled_input)

    # Make class predictions.
    label_class = xgb_model.predict(scaled_input)

    if label_class == 0:
        return class_proba[0], "Hip-Hop"

    elif label_class == 1:
        return class_proba[0], "Rock"



