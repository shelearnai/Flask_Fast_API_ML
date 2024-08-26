from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# Load dataset and split into train/test sets
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save the model

with open('model.pkl','wb') as file:
    pickle.dump(model, file)
