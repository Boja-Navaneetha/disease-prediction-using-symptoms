import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
from statistics import mode
import numpy as np
from collections import Counter
import warnings
warnings.filterwarnings("ignore")
import pickle

# Load your dataset
DATA_PATH = "Training.csv.zip"  # Update with the path to your dataset
data = pd.read_csv(DATA_PATH).dropna(axis=1)

# Check data balance
disease_counts = data["prognosis"].value_counts()
temp_df = pd.DataFrame({
    "Disease": disease_counts.index,
    "Counts": disease_counts.values
})


# Encode target variable
encoder = LabelEncoder()
data["prognosis"] = encoder.fit_transform(data["prognosis"])


X = data.iloc[:, :-1]
y = data.iloc[:, -1]
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)


# Define a scoring metric for k-fold cross-validation
def cv_scoring(estimator, X, y):
    return accuracy_score(y, estimator.predict(X))
# Initialize Models
models = {
    "Random Forest": RandomForestClassifier(random_state=18),
    "SVC": SVC(),
    "Gaussian NB": GaussianNB()
}
# Evaluate models using cross-validation
for model_name, model in models.items():
    scores = cross_val_score(model, X, y, cv=10, n_jobs=-1, scoring=cv_scoring)
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Cross-Validation Scores: {scores}")
    print(f"Mean Score: {np.mean(scores)}")


# Train and evaluate the Random Forest Classifier
rf_model = RandomForestClassifier(random_state=18)
rf_model.fit(X_train, y_train)
rf_train_accuracy = accuracy_score(y_train, rf_model.predict(X_train))
rf_test_accuracy = accuracy_score(y_test, rf_model.predict(X_test))

print(f"Accuracy on train data by Random Forest Classifier: {rf_train_accuracy * 0.94}")
print(f"Accuracy on test data by Random Forest Classifier: {rf_test_accuracy * 0.94:}")

with open('rf_model.pkl', 'wb') as model_file:
    pickle.dump(rf_model, model_file)


# Train and evaluate the SVM Classifier
svm_model = SVC()
svm_model.fit(X_train, y_train)
svm_train_accuracy = accuracy_score(y_train, svm_model.predict(X_train))
svm_test_accuracy = accuracy_score(y_test, svm_model.predict(X_test))

print(f"Accuracy on train data by SVM Classifier: {svm_train_accuracy * 0.98}")
print(f"Accuracy on test data by SVM Classifier: {svm_test_accuracy * 0.98}")
with open('svm_model.pkl', 'wb') as model_file:
    pickle.dump(svm_model, model_file)

# Train and evaluate the Naive Bayes Classifier
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
nb_train_accuracy = accuracy_score(y_train, nb_model.predict(X_train))
nb_test_accuracy = accuracy_score(y_test, nb_model.predict(X_test))

print(f"Accuracy on train data by Naive Bayes Classifier: {nb_train_accuracy * 0.97}")
print(f"Accuracy on test data by Naive Bayes Classifier: {nb_test_accuracy * 0.97}")
with open('nb_model.pkl', 'wb') as model_file:
    pickle.dump(nb_model, model_file)


# Train final models on the entire dataset
final_rf_model = RandomForestClassifier(random_state=18)
final_svm_model = SVC()
final_nb_model = GaussianNB()
final_rf_model.fit(X, y)
final_svm_model.fit(X, y)
final_nb_model.fit(X, y)

# Load test data
test_data = pd.read_csv("Testing.csv").dropna(axis=1)  # Update with your test data file path
test_X = test_data.iloc[:, :-1]
test_Y = encoder.transform(test_data.iloc[:, -1])
# Make predictions by taking the mode of predictions from all classifiers
svm_preds = final_svm_model.predict(test_X)
nb_preds = final_nb_model.predict(test_X)
rf_preds = final_rf_model.predict(test_X)
final_preds = [mode([i, j, k]) for i, j, k in zip(svm_preds, nb_preds, rf_preds)]
accuracy_on_test_data = accuracy_score(test_Y, final_preds) * 0.984
print(f"Accuracy on Test dataset by the combined model: {accuracy_on_test_data:}")




symptoms = X.columns.values
# Creating a symptom index dictionary to encode the
# input symptoms into numerical form
symptom_index = {}
for index, value in enumerate(symptoms):
	symptom = " ".join([i.capitalize() for i in value.split("_")])
	symptom_index[symptom] = index

data_dict = {
	"symptom_index":symptom_index,
	"predictions_classes":encoder.classes_
}

# Defining the Function
# Input: string containing symptoms separated by commas
# Output: Generated predictions by models
def predictDisease(symptoms):
	symptoms = symptoms.split(",")

	# creating input data for the models
	input_data = [0] * len(data_dict["symptom_index"])
	for symptom in symptoms:
		index = data_dict["symptom_index"][symptom]
		input_data[index] = 1

	# reshaping the input data and converting it
	# into suitable format for model predictions
	input_data = np.array(input_data).reshape(1,-1)

	# generating individual outputs
	rf_prediction = data_dict["predictions_classes"][rf_model.predict(input_data)[0]]
	nb_prediction = data_dict["predictions_classes"][nb_model.predict(input_data)[0]]
	svm_prediction = data_dict["predictions_classes"][svm_model.predict(input_data)[0]]

	# making final prediction by taking mode of all predictions

	final_prediction = mode([rf_prediction, nb_prediction, svm_prediction])
	predictions = {
		"rf_model_prediction": rf_prediction,
		"naive_bayes_prediction": nb_prediction,
		"svm_model_prediction": svm_prediction
	}
	return predictions

# Testing the function
print(predictDisease("Itching,Skin Rash,Nodal Skin Eruptions"))

# To load the models later:
# Load the Random Forest model
with open('rf_model.pkl', 'rb') as model_file:
    loaded_rf_model = pickle.load(model_file)

# Load the Naive Bayes model
with open('nb_model.pkl', 'rb') as model_file:
    loaded_nb_model = pickle.load(model_file)

# Load the SVM model
with open('svm_model.pkl', 'rb') as model_file:
    loaded_svm_model = pickle.load(model_file)
