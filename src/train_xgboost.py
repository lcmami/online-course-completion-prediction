import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.special import expit
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier


# =========================
# 0. 建立資料夾（避免存檔報錯）
# =========================
os.makedirs("data", exist_ok=True)
os.makedirs("images", exist_ok=True)


# =========================
# 1. 產生模擬資料
# =========================
np.random.seed(23)
n = 300

video_count = np.random.poisson(lam=10, size=n)
quiz_count = np.random.poisson(lam=5, size=n)
login_days = np.random.poisson(lam=7, size=n)

# 隱藏的線性邏輯機率模型
logit = (
    -4.0
    + 0.18 * video_count
    + 0.30 * quiz_count
    + 0.22 * login_days
)

# sigmoid 轉機率
prob = expit(logit)

# 根據機率產生 completed 標籤（0/1）
completed = np.random.binomial(1, prob)

# 建立 DataFrame
df = pd.DataFrame({
    "video_count": video_count,
    "quiz_count": quiz_count,
    "login_days": login_days,
    "completed": completed
})

print("=== Simulated Data Preview ===")
print(df.head())

# 存成 CSV
df.to_csv("data/simulated_data.csv", index=False)
print("\n資料已儲存到：data/simulated_data.csv")


# =========================
# 2. 資料切分
# =========================
X = df[["video_count", "quiz_count", "login_days"]]
y = df["completed"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=23
)

print(f"\n訓練集筆數：{len(X_train)}")
print(f"測試集筆數：{len(X_test)}")


# =========================
# 3. 建立並訓練 XGBoost 模型
# =========================
model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=23
)

model.fit(X_train, y_train)


# =========================
# 4. 模型預測與評估
# =========================
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.3f}")


# =========================
# 5. 特徵重要度分析
# =========================
importances = model.feature_importances_

importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": importances
}).sort_values(by="importance", ascending=False)

print("\n=== Feature Importance ===")
print(importance_df)

# 畫圖
plt.figure(figsize=(8, 5))
plt.bar(importance_df["feature"], importance_df["importance"])
plt.title("XGBoost Feature Importance")
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig("images/feature_importance.png")
plt.show()

print("\n特徵重要度圖已儲存到：images/feature_importance.png")