import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import xgboost as xgb  # Import XGBoost
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score, roc_auc_score, f1_score, recall_score, make_scorer, auc, roc_curve, precision_recall_curve, average_precision_score
from sklearn.preprocessing import LabelEncoder
import joblib

# Import the csv file.
echo_tracks = pd.read_csv("genre_dataset.csv")

# Create features(predictor) variables.
features = echo_tracks.drop(["genre_top", "track_id"], axis=1)

# Create labels(outcome).
labels = echo_tracks.genre_top

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

# Predict the test outcomes.
# val_preds = xgb_model.predict(scaled_test_features)

# # Print the confusion_matrix.
# conf_matrx = confusion_matrix(test_labels, val_preds)
# print(classification_report(test_labels, val_preds))

# sns.heatmap(data=conf_matrx, annot=True, fmt='.0f')
# plt.xlabel("Predicted values")
# plt.ylabel("Actual values")

# print()


# # Convert the target numpy variable into a dataframe
# encoded_labels = pd.DataFrame(encoded_labels, columns=["genre"])



# # Train and predict using StratifiedKFold.
# skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=20202024)
# # Define empty evaluation lists.
# accuracy_train, precision_train, recall_train, f1_train, roc_auc_train, confusion_matrices_train = [], [], [], [], [], []
# accuracy_val, precision_val, recall_val, f1_val, roc_auc_val, confusion_matrices_val = [], [], [], [], [], []

# for fold, (train_index, val_index) in enumerate(skf.split(features, encoded_labels)):
#     X_train, X_val = features.iloc[train_index], features.iloc[val_index]
#     y_train, y_val = encoded_labels.iloc[train_index], encoded_labels.iloc[val_index]

#     xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=False)
#     # Predict the train labels.
#     train_pred = xgb_model.predict(X_train)  # Train set.
#     val_pred = xgb_model.predict(X_val)   # Validation set.

#     # Evaluation metrics for Training sets.
#     train_accuracy = accuracy_score(y_train, train_pred)
#     train_precision = precision_score(y_train, train_pred)
#     train_recall = recall_score(y_train, train_pred)
#     train_f1 = f1_score(y_train, train_pred)
#     train_roc_auc = roc_auc_score(y_train, xgb_model.predict_proba(X_train)[:, 1])
#     train_confusion_matrix = confusion_matrix(y_train, train_pred)

#     # Evaluation metrics for validation sets.
#     val_accuracy = accuracy_score(y_val, val_pred)
#     val_precision = precision_score(y_val, val_pred)
#     val_recall = recall_score(y_val, val_pred)
#     val_f1 = f1_score(y_val, val_pred)
#     val_roc_auc = roc_auc_score(y_val, xgb_model.predict_proba(X_val)[:, 1])
#     val_confusion_matrix = confusion_matrix(y_val, val_pred)

#     # Append metrics to the initialized lists.
#     # Training sets.
#     accuracy_train.append(train_accuracy)
#     precision_train.append(train_precision)
#     recall_train.append(train_recall)
#     f1_train.append(train_f1)
#     roc_auc_train.append(train_roc_auc)
#     confusion_matrices_train.append(train_confusion_matrix)

#     # Validation sets.
#     accuracy_val.append(val_accuracy)
#     precision_val.append(val_precision)
#     recall_val.append(val_recall)
#     f1_val.append(val_f1)
#     roc_auc_val.append(val_roc_auc)
#     confusion_matrices_val.append(val_confusion_matrix)


# # Compute the mean of the metrics.

# # accuracy.
# mean_accuracy_train = np.mean(accuracy_train)
# mean_accuracy_val= np.mean(accuracy_val)

# # precision.
# mean_precision_train = np.mean(precision_train)
# mean_precision_val= np.mean(precision_val)

# # recall.
# mean_recall_train = np.mean(recall_train)
# mean_recall_val= np.mean(recall_val)

# # f1 score.
# mean_f1_train = np.mean(f1_train)
# mean_f1_val= np.mean(f1_val)

# # roc auc score.
# mean_roc_auc_train = np.mean(roc_auc_train)
# mean_roc_auc_val = np.mean(roc_auc_val)




# # Display evaluation metrics.
# print("Mean Accuracy Train", mean_accuracy_train)
# print("Mean Accuracy Validation", mean_accuracy_val)
# print()

# print("Mean Precision Train", mean_precision_train)
# print("Mean Precision Validation", mean_precision_val)
# print()

# print("Mean Recall Train", mean_recall_train)
# print("Mean Recall Validation", mean_recall_val)
# print()

# print("Mean ROC AUC Train", mean_roc_auc_train)
# print("Mean ROC AUC Validation", mean_roc_auc_val)
# print()

# print("Mean F1 Score Train", mean_f1_train)
# print("Mean F1 Score Validation", mean_f1_val)
# print()




# # Save the best model to a joblib file.
# joblib.dump(xgb_model, 'best_xgb_model.joblib')

# # Load the saved model in the joblib file.
# loaded_model = joblib.load('best_xgb_model.joblib')





# def accept_user_input():
#     """This function accepts user inputs, that it accepts necessary features input which is later used for prediction."""
#     feature_1 = float(input("Enter acousticness: " ))
#     feature_2 = float(input("Enter danceability: "))
#     feature_3 = float(input("Enter energy: "))
#     feature_4 = float(input("Enter instrumentalness: "))
#     feature_5 = float(input("Enter liveness: "))
#     feature_6 = float(input("Enter speechiness: "))
#     feature_7 = float(input("Enter tempo: "))
#     feature_8 = float(input("Enter valence: "))

#     return feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8






def predict_class_proba(user_input):
    "A function that predict the probabilities of each classes(positive(Rock) and negative(Hip Hop)) based on the collected input features."

    # # Extract the user input from the accept_user_input function.
    # f1, f2, f3, f4, f5, f6, f7, f8 = user_input()


    # user_input = np.array([f1, f2, f3, f4, f5, f6, f7, f8]).reshape(-1, 8)
    user_input = np.array([user_input]).reshape(-1, 8)

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



