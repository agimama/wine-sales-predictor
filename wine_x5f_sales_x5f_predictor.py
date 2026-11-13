"""
Wine Shop Sales Predictor - Personal Portfolio Project
By Agnes Chiamaka Chinedu | AI & Machine Learning Engineer

What it does:
Predicts daily wine sales (in bottles) for a small wine shop like Bottle O'clock,
using features a real shop actually tracks: day of week, promotions, price, stock, weather.

Skills shown:
- Linear Regression, train-test splitting, StandardScaler
- Data cleaning, feature engineering, model evaluation
- Turning business data into a useful prediction

How to run in Google Colab:
1. Upload this file + wine_shop_sales.csv to Colab, or copy the code into a new notebook
2. Run all cells - it will print the results and save a chart

No real customer data is used - the dataset is synthetic (made-up but realistic).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# --- 1. Load data ---
# In Colab: upload wine_shop_sales.csv first, or run the dataset generator below
try:
    df = pd.read_csv("wine_shop_sales.csv")
except FileNotFoundError:
    print("wine_shop_sales.csv not found - generating a sample dataset...")
    np.random.seed(42)
    n = 365
    df = pd.DataFrame({
        "day_of_week": np.random.randint(0, 7, n),          # 0=Mon ... 6=Sun
        "is_weekend": np.random.randint(0, 2, n),
        "is_promo": np.random.randint(0, 2, n),             # 1 = promotion running
        "avg_price": np.random.uniform(8000, 25000, n),     # Naira per bottle
        "stock_bottles": np.random.randint(20, 300, n),
        "temp_c": np.random.uniform(24, 34, n),
        "instagram_posts": np.random.randint(0, 4, n),      # marketing effort that day
    })
    # Realistic pattern: weekends + promos + posts = more sales, high price = fewer sales
    df["sales_bottles"] = (
        18 + df["is_weekend"]*22 + df["is_promo"]*15 + df["instagram_posts"]*6
        - (df["avg_price"]-8000)/2500 + df["stock_bottles"]*0.05
        + np.random.normal(0, 6, n)
    ).clip(lower=2).round().astype(int)

print("Dataset shape:", df.shape)
print(df.head())
print("\nSales summary:")
print(df["sales_bottles"].describe())

# --- 2. Features & target ---
features = ["day_of_week", "is_weekend", "is_promo", "avg_price", "stock_bottles", "temp_c", "instagram_posts"]
X = df[features]
y = df["sales_bottles"]

# --- 3. Train-test split (80% train, 20% test) ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTrain: {len(X_train)} days | Test: {len(X_test)} days")

# --- 4. Standardize with StandardScaler (fit ONLY on train - no data leakage) ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 5. Train Linear Regression ---
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# --- 6. Evaluate ---
y_pred = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\n=== RESULTS ===")
print(f"Mean Absolute Error: {mae:.2f} bottles  (on average, prediction is off by this much)")
print(f"R2 Score: {r2:.2f}  (1.0 = perfect, 0 = no better than guessing the average)")

# --- 7. What drives sales? (feature importance) ---
importance = pd.DataFrame({"feature": features, "coefficient": model.coef_}).sort_values("coefficient", ascending=False)
print("\nWhat drives sales most:")
print(importance)

# --- 8. Try a real prediction ---
# Example: Saturday, promo ON, price 12000, 150 bottles in stock, 30C, 2 Instagram posts
example = pd.DataFrame([[5, 1, 1, 12000, 150, 30, 2]], columns=features)
example_scaled = scaler.transform(example)
prediction = model.predict(example_scaled)[0]
print(f"\nExample: Saturday + promo + 2 posts -> Predicted sales: {prediction:.0f} bottles")

# --- 9. Chart: Actual vs Predicted ---
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", label="Perfect prediction")
plt.xlabel("Actual sales (bottles)")
plt.ylabel("Predicted sales (bottles)")
plt.title("Wine Shop Sales - Actual vs Predicted (Linear Regression)")
plt.legend()
plt.tight_layout()
plt.savefig("actual_vs_predicted.png")
plt.show()
print("\nChart saved as actual_vs_predicted.png")
print("Done! You can now explain: problem, data, model, result, and what you would improve next.")
