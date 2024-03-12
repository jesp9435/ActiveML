import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load the Fashion MNIST dataset
mnist = fetch_openml(name="Fashion-MNIST", version=1)
X, y = mnist.data, mnist.target

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a committee of random forest classifiers
num_models = 5
committee = [RandomForestClassifier(n_estimators=10) for _ in range(num_models)]

# Train each model in the committee
for model in committee:
    model.fit(X_train, y_train)

# Generate predictions from the committee members
committee_predictions = np.array([model.predict(X_test) for model in committee])

# Calculate the disagreement among the committee members
def disagreement(predictions):
    # Example disagreement metric: standard deviation of predictions
    return np.std(predictions, axis=0)

# Calculate the disagreement for the committee
committee_disagreement = disagreement(committee_predictions)

# Print the disagreement values
print("Disagreement values for each test input:")
for i, d in enumerate(committee_disagreement):
    print(f"Input {i+1}: {d:.4f}")

# Now you can use the disagreement values to determine uncertainty.
# For example, you can rank the test inputs based on their disagreement values and select the most uncertain ones.

# Feel free to explore other disagreement metrics and investigate the impact of the number of committee members.
# You can also compare this approach to other uncertainty estimation methods.

# Note: In practice, use more sophisticated models and real-world data for better results.
