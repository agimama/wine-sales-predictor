# Wine Shop Sales Predictor 🍷

**By Agnes Chiamaka Chinedu — AI & Machine Learning Engineer**

## The Problem
As Head of Marketing at a wine shop, I kept asking: *how many bottles will we sell tomorrow?*
If we know that, we can stock enough, plan promos, and avoid running out on a busy Saturday.

This project answers that question with Machine Learning.

## What I Built
A Linear Regression model that predicts daily wine sales (in bottles) from 7 features a real shop tracks:
- day_of_week, is_weekend
- is_promo (is a promotion running?)
- avg_price, stock_bottles
- temp_c, instagram_posts (marketing effort that day)

## What I Did
1. Created a realistic 365-day dataset (synthetic — no real customer data)
2. Split data into train (80%) / test (20%) with `train_test_split`
3. Standardized features with `StandardScaler` — fitted ONLY on train data to avoid leakage
4. Trained a `LinearRegression` model
5. Evaluated with MAE and R², and checked which features drive sales

## Results
- **Mean Absolute Error: 4.73 bottles** — on average, prediction is off by ~5 bottles
- **R² Score: 0.88** — model explains 88% of the variation in sales
- **Biggest drivers:** weekends, promos, and Instagram posts increase sales. Higher price decreases sales.
- **Example:** Saturday + promo + 2 Instagram posts → predicts ~73 bottles

See chart: `actual_vs_predicted.png`

## Tools
Python, pandas, NumPy, scikit-learn (LinearRegression, StandardScaler, train_test_split), Matplotlib, Google Colab

## How to Run in Google Colab
1. Go to colab.research.google.com → New Notebook
2. Upload `wine_shop_sales.csv` and `wine_sales_predictor.py` (or copy the code into cells)
3. Run all — you will see the results + chart

## What I Would Improve Next
- Try Logistic Regression to predict "busy day vs quiet day" (classification)
- Try Random Forest to capture non-linear patterns
- Use real shop data (with permission) instead of synthetic data

## Files
- `wine_sales_predictor.py` — full project
- `generate_dataset.py` — creates the synthetic dataset
- `wine_shop_sales.csv` — 365 days of data
- `actual_vs_predicted.png` — results chart


