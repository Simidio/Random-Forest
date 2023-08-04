import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Create an example dataset with 1000 observations
data = pd.DataFrame({
    'revenue': [5000000, 1200000, 800000, 650000, 1400000] * 200,
    'num_employees': [50, 18, 12, 10, 25] * 200,
    'sector': ['Technology', 'Construction', 'Commerce', 'Services', 'Manufacturing'] * 200,
    'failure_rate': [0.2, 0.8, 0.6, 0.5, 0.7] * 200
})

# Encode the "sector" variable as binary variables
data = pd.get_dummies(data, columns=['sector'])

# Define a failure threshold
threshold = 0.5

# Convert the "failure_rate" variable into a binary variable
data['failure'] = (data['failure_rate'] >= threshold).astype(int)
data = data.drop('failure_rate', axis=1)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data.drop('failure', axis=1), data['failure'], test_size=0.2, random_state=42)

# Create the Random Forest model with 500 trees
rf = RandomForestClassifier(n_estimators=500, random_state=42)

# Train the model on the training set
rf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf.predict(X_test)

# Evaluate the performance of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))

# Print the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:")
print(cm)

# Print the precision, recall, and f1-score
print("Classification report:")
print(classification_report(y_test, y_pred))

# Plot all the trees in the Random Forest model
#plt.figure(figsize=(20,10))
#for tree in rf.estimators_:
#    plot_tree(tree, feature_names=X_train.columns.tolist(), filled=True)
#    plt.show()

# Load the test data
new_data = pd.DataFrame({
    'revenue': [4000000, 900000, 700000, 600000, 1300000],
    'num_employees': [40, 15, 9, 8, 20],
    'sector': ['Technology', 'Construction', 'Commerce', 'Services', 'Manufacturing'],
})

# Encode the "sector" variable as binary variables
new_data = pd.get_dummies(new_data, columns=['sector'])

# Make predictions on the new observations
y_pred = rf.predict(new_data)

# Print the predictions
print("Predictions on failure rate:")
print(y_pred)
