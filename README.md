# Linear Regression Optimizer

A from-scratch implementation of three gradient descent variants for linear regression, built using only NumPy and Pandas — no ML libraries. Applied to the [UCI Auto MPG dataset](https://archive.ics.uci.edu/dataset/9/auto+mpg) to predict vehicle fuel efficiency (mpg).

---

## Optimizers Implemented

| Method | Gradient Computed On | Updates per Epoch |
|---|---|---|
| Batch Gradient Descent (BGD) | Full dataset | 1 |
| Stochastic Gradient Descent (SGD) | 1 random sample | n |
| Mini-Batch Gradient Descent (MBGD) | 32-sample batches | n / 32 |

---

## Dataset

**[UCI Auto MPG](https://archive.ics.uci.edu/dataset/9/auto+mpg)** — originally from the CMU StatLib library, donated in 1993 by R. Quinlan.

The dataset contains **398 instances** (392 after dropping missing values) of cars from the 1970s–80s. The goal is to predict city-cycle fuel consumption in **miles per gallon (mpg)** from 7 features:

| Feature | Type | Description |
|---|---|---|
| cylinders | Integer | Number of engine cylinders |
| displacement | Continuous | Engine displacement (cubic inches) |
| horsepower | Continuous | Engine horsepower *(has missing values)* |
| weight | Continuous | Vehicle weight (lbs) |
| acceleration | Continuous | 0–60 mph time (seconds) |
| model year | Integer | Model year (70–82) |
| origin | Categorical | 1 = USA, 2 = Europe, 3 = Japan |

> **Citation:** Quinlan, R. (1993). Auto MPG [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5859H

---

## How It Works

**Preprocessing pipeline** (applied after loading the dataset above):
- Dropped 6 rows with missing values (~1.5% of data)
- One-hot encoded `origin` (categorical: USA, Europe, Japan) with `drop_first=True` to avoid multicollinearity
- 80/20 train/test split with random shuffling
- Z-score normalization computed on training data only — applied to test set to prevent data leakage

**Training:**
- Bias term added via a column of ones (intercept learned automatically)
- All variants use early stopping: training halts when RSS improvement drops below `tol=1e-6`
- Gradient normalized by number of samples so learning rate stays stable regardless of dataset size

**Evaluation metric:** Residual Sum of Squares (RSS) on held-out test data

---

## Sample Output

```
Training BGD
  BGD converged at epoch 312
Training SGD
  SGD converged at epoch 87
Training Mini-Batch GD
  Mini-Batch GD converged at epoch 204

Method             Train RSS         Test RSS
BGD                  1821.43           487.92
SGD                  1893.17           512.38
Mini-Batch GD        1849.61           498.74
```
> Note: Results vary slightly each run due to random shuffling. Replace with your actual output.

---

## Project Structure

```
├── main.py        # Full implementation: preprocessing, training, evaluation
└── README.md
```

---

## Setup & Run

**Requirements:**
```bash
pip install ucimlrepo pandas numpy
```

**Run:**
```bash
python main.py
```

---

## Key Design Decisions

- **No sklearn or ML frameworks** — all math implemented directly in NumPy to demonstrate understanding of the underlying algorithms
- **Data leakage prevention** — normalization stats derived from training set only, then applied to test set
- **Shuffling in SGD** — data reshuffled each epoch to avoid the model learning sample-ordering patterns
- **Normalized gradient in BGD/MBGD** — dividing by batch size decouples the learning rate from dataset size

---

## Author

Andres Tenesaca (ast7673)