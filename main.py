from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
"""
Andres Tenesaca
"""
auto_mpg = fetch_ucirepo(id=9)
X_raw = auto_mpg.data.features
y_raw = auto_mpg.data.targets

# Preprocessing
# Assumption: rows with missing values are dropped rather than imputed.
# Only 6 rows (~1.5% of data) are missing, so dropping has negligible impact.
df = pd.concat([X_raw, y_raw], axis=1).dropna()

# Assumption: 'origin' is treated as categorical (not ordinal).
# drop_first=True drops origin_1 (USA) as the reference category to avoid multicollinearity.
X = pd.get_dummies(df.drop(columns=['mpg']), columns=['origin'], drop_first=True)
X = X.values.astype(float)
y = df['mpg'].values.astype(float)

# Train/test split 80/20
n = len(X)
idx = np.random.permutation(n)
split = int(0.8 * n)
X_train, X_test = X[idx[:split]], X[idx[split:]]
y_train, y_test = y[idx[:split]], y[idx[split:]]

# Assumption: scaling parameters are computed on training data only to prevent data leakage.
# The same mean/std is then applied to the test set.
mean = X_train.mean(axis=0)
std  = X_train.std(axis=0)
std[std == 0] = 1
X_train = (X_train - mean) / std
X_test  = (X_test  - mean) / std

# Add bias column (column of 1s so intercept is learned automatically)
X_train = np.c_[np.ones(len(X_train)), X_train]
X_test  = np.c_[np.ones(len(X_test)),  X_test]

#HELPER FUNCTIONS
def compute_rss(X, y, beta):
    residuals = y - X @ beta
    return np.sum(residuals ** 2)

def compute_gradient(X, y, beta):
    return -2 * X.T @ (y - X @ beta)


# GRADIENT DESCENT VARIANTS
def batch_gradient_descent(X, y, lr=0.05, epochs=2000, tol=1e-6):
    """
    BGD: Uses ALL samples to compute gradient each epoch.
    One update per epoch.
    """
    n, p = X.shape
    beta = np.zeros(p)

    for epoch in range(epochs):
        prev_rss = compute_rss(X, y, beta)
        grad = compute_gradient(X, y, beta) / n # Assumption: gradient is normalized by number of samples so the 
                                                # learning rate is not sensitive to dataset size.
        beta -= lr * grad
        if abs(prev_rss - compute_rss(X, y, beta)) < tol:
            print(f"  BGD converged at epoch {epoch}")
            break

    return beta


def stochastic_gradient_descent(X, y, lr=0.001, epochs=200, tol=1e-6):
    """
    SGD: Uses ONE random sample per update.
    n updates per epoch.
    """
    n, p = X.shape
    beta = np.zeros(p)
    prev_rss = float('inf')

    for epoch in range(epochs):
        for i in np.random.permutation(n):       # Assumption: data is shuffled every epoch to prevent the model from learning
                                                 # patterns in the ordering of samples.
            xi, yi = X[i], y[i]
            grad = -2 * xi * (yi - xi @ beta)    # gradient for one sample
            beta -= lr * grad

        rss = compute_rss(X, y, beta)
        if abs(prev_rss - rss) < tol:
            print(f"  SGD converged at epoch {epoch}")
            break
        prev_rss = rss

    return beta


def minibatch_gradient_descent(X, y, lr=0.05, epochs=500, batch_size=32, tol=1e-6):
    """
    Mini-Batch GD: Uses a small batch per update.
    """
    # Assumption: batch_size=32 is chosen as a standard default. Larger batches
    # give smoother gradients; smaller batches introduce more noise but update more frequently.
    n, p = X.shape
    beta = np.zeros(p)
    prev_rss = float('inf')

    for epoch in range(epochs):
        perm = np.random.permutation(n)
        for start in range(0, n, batch_size):
            Xb = X[perm[start:start+batch_size]]
            yb = y[perm[start:start+batch_size]]
            grad = compute_gradient(Xb, yb, beta) / len(Xb)
            beta -= lr * grad

        rss = compute_rss(X, y, beta)
        if abs(prev_rss - rss) < tol:
            print(f"  Mini-Batch GD converged at epoch {epoch}")
            break
        prev_rss = rss

    return beta


# Train all three methods
print("Training BGD")
beta_bgd  = batch_gradient_descent(X_train, y_train)

print("Training SGD")
beta_sgd  = stochastic_gradient_descent(X_train, y_train)

print("Training Mini-Batch GD")
beta_mbgd = minibatch_gradient_descent(X_train, y_train)


# Evaluate and print results
for name, beta in [("BGD", beta_bgd), ("SGD", beta_sgd), ("Mini-Batch GD", beta_mbgd)]:
    train_rss = compute_rss(X_train, y_train, beta)
    test_rss  = compute_rss(X_test,  y_test,  beta)
    print(f"{name:<18} {train_rss:>16.2f} {test_rss:>16.2f}")