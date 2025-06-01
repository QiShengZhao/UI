import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy import stats

# 加载保存的模型和缩放器
print("加载模型和数据...")
try:
    # 尝试加载中文版本的模型
    model = joblib.load('best_rf_model_chinese.pkl')
    scaler = joblib.load('rf_feature_scaler_chinese.pkl')
    model_version = "中文版"
except FileNotFoundError:
    try:
        # 如果中文版本不存在，尝试加载英文版本
        model = joblib.load('best_rf_model.pkl')
        scaler = joblib.load('rf_feature_scaler.pkl')
        model_version = "英文版"
    except FileNotFoundError:
        print("错误：找不到已训练的模型文件。请先运行训练脚本。")
        exit(1)

# 读取原始数据
df = pd.read_csv('test1.csv')

# 计算韧性（目标变量）
df['韧性'] = df[['实测1', '实测2', '实测3']].mean(axis=1)

# 清理数据
df_clean = df.dropna()

# 移除异常值，保持与训练时相同的预处理
z_scores = stats.zscore(df_clean['韧性'])
abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3)
df_no_outliers = df_clean[filtered_entries]

# 选择特征（与训练时相同）
feature_columns = [col for col in df_no_outliers.columns if col not in ['韧性', '实测1', '实测2', '实测3']]
X = df_no_outliers[feature_columns]
y = df_no_outliers['韧性']

# 创建训练集和测试集（为了获得测试样本）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 缩放特征
X_test_scaled = scaler.transform(X_test)

# 对测试集进行预测
predictions = model.predict(X_test_scaled)

# 计算误差
errors = y_test.values - predictions
abs_errors = np.abs(errors)

# 按误差大小排序
sorted_indices = np.argsort(abs_errors)
test_indices = np.array(y_test.index)

# 选择不同误差范围的样本
# 1-2. 低误差样本
low_error_indices = sorted_indices[:2]
# 3. 中等误差样本
mid_error_index = sorted_indices[len(sorted_indices)//2:len(sorted_indices)//2+1]
# 4-5. 高误差样本
high_error_indices = sorted_indices[-2:]

# 组合样本索引
sample_indices = np.concatenate([low_error_indices, mid_error_index, high_error_indices])

print(f"\n使用{model_version}随机森林模型的5个预测示例：")
print("序号 | 实际韧性值 | 预测韧性值 | 绝对误差 | 相对误差(%)")
print("-" * 65)

for i, idx in enumerate(sample_indices):
    test_idx = test_indices[idx]
    actual = y_test.loc[test_idx]
    predicted = predictions[idx]
    abs_error = abs(actual - predicted)
    rel_error = (abs_error / actual) * 100 if actual != 0 else float('inf')
    
    # 选择误差级别
    if i < 2:
        error_level = "低误差"
    elif i == 2:
        error_level = "中等误差"
    else:
        error_level = "高误差"
    
    print(f"{i+1:2d} | {actual:9.2f}  | {predicted:9.2f}  | {abs_error:7.2f} | {rel_error:8.2f} | {error_level}")

# 计算总体指标
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)

print("\n测试集性能指标:")
print(f"测试样本数: {len(y_test)}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.4f}")
print(f"MAE: {mae:.2f}") 