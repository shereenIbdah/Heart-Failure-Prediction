# Jehad Hamayel 1200348
# Shereen Ibdah 1200373
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC

# Read data from CSV file
dataSet = pd.read_csv('DataSet/heart.csv')
# Split columns between features and target
features = dataSet.drop('HeartDisease', axis=1)
target = dataSet['HeartDisease']
# Define categorical and numerical columns
categoricalColumns = features.select_dtypes(include=['category', 'object']).columns
numericalColumns = features.select_dtypes(include=['int64', 'float64']).columns

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 8))  # Adjusting the layout to fit all subplots

for ax, feature in zip(axes.flatten(), numericalColumns):
    dataSet.boxplot(column=feature, by='HeartDisease', ax=ax)
    ax.set_title(f'Distribution of {feature} by HeartDisease')
    ax.set_xlabel('HeartDisease')
    ax.set_ylabel(feature)
plt.tight_layout()
plt.suptitle('Boxplots of Features by Heart Disease Class', fontsize=16, y=1.02)
plt.show()
# Print the head of the data file
print(dataSet.head())

# Convert categorical columns to string data type
dataSet[categoricalColumns] = dataSet[categoricalColumns].astype("string")
dataSet.info()

# Print the description of the file in terms of statistics
description = dataSet.describe(include='all').T
print(description)

# Drawing Box Plots, Pie Charts and Histograms for featuers
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(12, 8))
axes = axes.flatten()
pie_dataset = pd.DataFrame(dataSet.groupby("Sex")["Sex"].count())
axes[0].pie(pie_dataset['Sex'], labels=pie_dataset.index, autopct='%1.1f%%', startangle=90)
axes[0].set_title('Gender')
pie_dataset = pd.DataFrame(dataSet.groupby("ChestPainType")["ChestPainType"].count())
axes[1].pie(pie_dataset['ChestPainType'], labels=pie_dataset.index, autopct='%1.1f%%', startangle=90)
axes[1].set_title('Chest Pain Type')
pie_dataset = pd.DataFrame(dataSet.groupby("ExerciseAngina")["ExerciseAngina"].count())
axes[2].pie(pie_dataset['ExerciseAngina'], labels=pie_dataset.index, autopct='%1.1f%%', startangle=90)
axes[2].set_title('Exercise Angina')
pie_dataset = pd.DataFrame(dataSet.groupby("HeartDisease")["HeartDisease"].count())
axes[3].pie(pie_dataset['HeartDisease'], labels=pie_dataset.index, autopct='%1.1f%%', startangle=90)
axes[3].set_title('Heart Disease')
pie_dataset = pd.DataFrame(dataSet.groupby("RestingECG")["RestingECG"].count())
axes[4].pie(pie_dataset['RestingECG'], labels=pie_dataset.index, autopct='%1.1f%%', startangle=90)
axes[4].set_title('Resting ECG')

pie_dataset = pd.DataFrame(dataSet.groupby("ST_Slope")["ST_Slope"].count())
axes[5].pie(pie_dataset['ST_Slope'], labels=pie_dataset.index, autopct='%1.1f%%', startangle=90)
axes[5].set_title('ST Slope')

palette = sns.color_palette("bright", len(numericalColumns))

sns.histplot(dataSet["Age"], kde=True, color=palette[0], ax=axes[6])
axes[6].set_title(f'Distribution of Age')

sns.histplot(dataSet["RestingBP"], kde=True, color=palette[1], ax=axes[7])
axes[7].set_title(f'Distribution of RestingBP')

sns.histplot(dataSet["Cholesterol"], kde=True, color=palette[2], ax=axes[8])
axes[8].set_title(f'Distribution of Cholesterol')

sns.histplot(dataSet["FastingBS"], kde=True, color=palette[3], ax=axes[9])
axes[9].set_title(f'Distribution of FastingBS')

sns.histplot(dataSet["MaxHR"], kde=True, color=palette[4], ax=axes[10])
axes[10].set_title(f'Distribution of MaxHR')

sns.histplot(dataSet["Oldpeak"], kde=True, color=palette[5], ax=axes[11])
axes[11].set_title(f'Distribution of Oldpeak')

plt.tight_layout()
plt.show()


missingValues = dataSet.isnull().sum()
print("Missing Values:\n", missingValues)

# Normalization by Z score for numerical Columns
scaler = StandardScaler()
ScaledNumericalData = scaler.fit_transform(features[numericalColumns])

# One Hot Encoded is applied to categorical features
OneHotEncodedData = OneHotEncoder(handle_unknown='ignore')
OneHotEncodedCategoricalData = OneHotEncodedData.fit_transform(features[categoricalColumns]).toarray()

featuresStack = np.hstack((ScaledNumericalData, OneHotEncodedCategoricalData))

# Divide the data into 80% training and 20% testing
trainingData, testingData, trainingTarget, testingTarget = train_test_split(featuresStack, target, test_size=0.2,
                                                                            random_state=10)
# Applying the knn Classifier With K = 1 to the data
knnClassifierWithK1 = KNeighborsClassifier(n_neighbors=1)
knnClassifierWithK1.fit(trainingData, trainingTarget)

predictedTargetWithK1 = knnClassifierWithK1.predict(testingData)

# Applying the knn Classifier With K = 3 to the data
knnClassifierWithK3 = KNeighborsClassifier(n_neighbors=3)
knnClassifierWithK3.fit(trainingData, trainingTarget)

predictedTargetWithK3 = knnClassifierWithK3.predict(testingData)

accuracyWithk1 = accuracy_score(testingTarget, predictedTargetWithK1)
accuracyWithk3 = accuracy_score(testingTarget, predictedTargetWithK3)
print("Accuracy of knn model with K=1 :", accuracyWithk1)
print("Accuracy of knn model with K=3 :", accuracyWithk3)


# Support Vector Machine (SVM)
SVM = SVC()

# Hyperparameters to test
svm_parameters = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'degree': [2, 3, 4, 5]}

# Grid Search with cross-validation
svm_grid_search = GridSearchCV(estimator=SVM, param_grid=svm_parameters, cv=5, scoring='accuracy', n_jobs=-1)
svm_grid_search.fit(trainingData, trainingTarget)

# Best parameters and score
svm_best_params = svm_grid_search.best_params_
svm_best_score = svm_grid_search.best_score_

print("Best Parameters for SVM:", svm_best_params)
print("Best Cross-Validation Score:", svm_best_score)

best_svm_model = svm_grid_search.best_estimator_
predictedTargetSVM = best_svm_model.predict(testingData)
accuracySVM = accuracy_score(testingTarget, predictedTargetSVM)
print("Accuracy of SVM :", accuracySVM)

confusionMatrixForsvm = confusion_matrix(testingTarget, predictedTargetSVM)

TP = confusionMatrixForsvm[1, 1]  # True Positives (actual = 1, predicted = 1)
FN = confusionMatrixForsvm[1, 0]  # False Negatives (actual = 1, predicted = 0)
FP = confusionMatrixForsvm[0, 1]  # False Positives (actual = 0, predicted = 1)
TN = confusionMatrixForsvm[0, 0]  # True Negatives (actual = 0, predicted = 0)

# Print the values
print("True Positives (TP):", TP)
print("False Negatives (FN):", FN)
print("False Positives (FP):", FP)
print("True Negatives (TN):", TN)

precision_svm = precision_score(testingTarget, predictedTargetSVM)
recall_svm = recall_score(testingTarget, predictedTargetSVM)
f1_score_svm = f1_score(testingTarget, predictedTargetSVM)

# Print the results
print("Precision of the best SVM model:", precision_svm)
print("Recall of the best SVM model:", recall_svm)
print("F1 Score of the best SVM model:", f1_score_svm)

results = svm_grid_search.cv_results_

# Print model information for model selection
print("Model Performance:\n")
for i in range(len(results["mean_test_score"])):
    print(f"Model {i + 1}:")
    print(f"    Parameters: {results['params'][i]}")
    print(f"    Mean Accuracy: {results['mean_test_score'][i]:.4f}")
    print()


# Random Forest
RFC = RandomForestClassifier()

# Hyperparameters to test
rfc_parameters = {"n_estimators": [10, 50, 100, 200], "max_depth": [5, 10, 20, 50]}

# Grid Search with cross-validation
RFC_classifier_grid_search = GridSearchCV(estimator=RFC, param_grid=rfc_parameters, cv=5, scoring='accuracy', n_jobs=-1)
RFC_classifier_grid_search.fit(trainingData, trainingTarget)

# Best parameters and score
RFC_best_params = RFC_classifier_grid_search.best_params_
RFC_best_score = RFC_classifier_grid_search.best_score_

print("Best Parameters for RFC:", RFC_best_params)
print("Best Cross-Validation Score:", RFC_best_score)

best_rfc_model = RFC_classifier_grid_search.best_estimator_
predictedTargetRFC = best_rfc_model.predict(testingData)
accuracyRFC = accuracy_score(testingTarget, predictedTargetRFC)

print("Accuracy of RFC :", accuracyRFC)

confusionMatrixForrfc = confusion_matrix(testingTarget, predictedTargetRFC)

TP = confusionMatrixForrfc[1, 1]  # True Positives (actual = 1, predicted = 1)
FN = confusionMatrixForrfc[1, 0]  # False Negatives (actual = 1, predicted = 0)
FP = confusionMatrixForrfc[0, 1]  # False Positives (actual = 0, predicted = 1)
TN = confusionMatrixForrfc[0, 0]  # True Negatives (actual = 0, predicted = 0)

# Print the values
print("True Positives (TP):", TP)
print("False Negatives (FN):", FN)
print("False Positives (FP):", FP)
print("True Negatives (TN):", TN)

precision_rfc = precision_score(testingTarget, predictedTargetRFC)
recall_rfc = recall_score(testingTarget, predictedTargetRFC)
f1_score_rfc = f1_score(testingTarget, predictedTargetRFC)

# Print the results
print("Precision of the best RFC model:", precision_rfc)
print("Recall of the best RFC model:", recall_rfc)
print("F1 Score of the best RFC model:", f1_score_rfc)

results = RFC_classifier_grid_search.cv_results_

# Print model information for model selection
print("Model Performance:\n")
for i in range(len(results["mean_test_score"])):
    print(f"Model {i + 1}:")
    print(f"    Parameters: {results['params'][i]}")
    print(f"    Mean Accuracy: {results['mean_test_score'][i]:.8f}")
    print()
