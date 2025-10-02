# ================================
# Vansh Gutgutia Logistic Regression 
# ================================

# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

# data
file_path = "StudentPerformanceFactors.csv"  
df = pd.read_csv(file_path)

# here is our yes/no value. is the score greater than the class avg?
avg_score = df["Exam_Score"].mean()
df["Target"] = (df["Exam_Score"] >= avg_score).astype(int)

# features
num_features = ["Hours_Studied", "Attendance", "Sleep_Hours",
                "Previous_Scores", "Tutoring_Sessions", "Physical_Activity"]

cat_features = ["Parental_Involvement", "Access_to_Resources", 
                "Motivation_Level", "Internet_Access", "Family_Income"]

# one hot encoder
df_encoded = pd.get_dummies(df[num_features + cat_features], drop_first=True)

# ensure all numeric
df_encoded = df_encoded.astype(float)

print("Total features after encoding:", df_encoded.shape[1])

# feature matrix
X = df_encoded.values.astype(float)
y = df["Target"].values.astype(float)

# scale 
scaler = RobustScaler()
X = scaler.fit_transform(X)

# test/train
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# bias
m_train = X_train.shape[0]
X_train = np.column_stack([np.ones(m_train), X_train])
m_test = X_test.shape[0]
X_test = np.column_stack([np.ones(m_test), X_test])

n_with_bias = X_train.shape[1]

# math
def sigmoid(z):
    z = np.array(z, dtype=float)  
    return 1 / (1 + np.exp(-z))

def predict_proba(X, w):
    return sigmoid(X @ w)

def binary_cross_entropy(y_true, y_prob, eps=1e-12):
    y_prob = np.clip(y_prob, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))

def gradient(X, y_true, y_prob):
    m = X.shape[0]
    return (1/m) * (X.T @ (y_prob - y_true))

# initialize params
w = np.zeros(n_with_bias)

# hyper params
learning_rate = 0.1
print("Learning Rate:", learning_rate)
num_iterations = 3000
print("Number of Iterations:", num_iterations)
cost_history = []

# initial cost
y_hat_init = predict_proba(X_train, w)
initial_cost = binary_cross_entropy(y_train, y_hat_init)
print("Initial parameters (w):", w)
print("Initial cost:", initial_cost)

# gradient descent
for i in range(num_iterations):
    y_hat = predict_proba(X_train, w)
    cost = binary_cross_entropy(y_train, y_hat)
    cost_history.append(cost)
    grad = gradient(X_train, y_train, y_hat)
    w -= learning_rate * grad

# final params and cost
final_cost = cost_history[-1]
print("Final parameters (w):", w)
print("Final cost:", final_cost)

# plot cost vs training
plt.figure()
plt.plot(range(len(cost_history)), cost_history)
plt.xlabel("Iteration")
plt.ylabel("Cost (Log-Loss)")
plt.title("Training: Cost vs. Iterations (Robust Scaled)")
plt.grid(True)
plt.show()

# plot cost vs parameters
param_indices = np.argsort(np.abs(w[1:]))[::-1][:3] + 1  # skip bias
print("Plotting cost sensitivity for parameter indices:", param_indices)

def compute_cost_given_w(mod_w):
    y_hat_mod = predict_proba(X_train, mod_w)
    return binary_cross_entropy(y_train, y_hat_mod)

for idx in param_indices:
    center = w[idx]
    sweep = np.linspace(center - 1.0, center + 1.0, 60)
    costs = []
    for val in sweep:
        w_tmp = w.copy()
        w_tmp[idx] = val
        costs.append(compute_cost_given_w(w_tmp))

    plt.figure()
    plt.plot(sweep, costs)
    plt.xlabel(f"Parameter w[{idx}]")
    plt.ylabel("Cost (Log-Loss)")
    plt.title(f"Cost vs Parameter w[{idx}] (holding others fixed)")
    plt.grid(True)
    plt.show()

# predict
def predict_label(X_new, w, threshold=0.5):
    return (predict_proba(X_new, w) >= threshold).astype(int)

preds = predict_label(X_test, w)
accuracy = (preds == y_test).mean()
print("Test accuracy:", accuracy)
