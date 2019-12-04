# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

# Data loading
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X = data.data       # descriptor
y = data.target     # target
X = X[:, :10]   # Choose 10 values from the front

# Learning
from sklearn.linear_model import LogisticRegression
model = LogisticRegression() # make instance of LogisticRegression class
model.fit(X, y) # leaning

# Prediction based on X
y_pred = model.predict(X)

# Evaluation
from sklearn.metrics import accuracy_score
score = accuracy_score(y, y_pred)
print("Accuracy score = %e" % score)