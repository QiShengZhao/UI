import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# 加载数据
print("加载数据...")
data = pd.read_csv('cleaned_data_no_noise.csv')

# 特征工程
print("执行特征工程...")

# 1. 创建新特征 - 化学成分比例和组合
data['C_Mn_Ratio'] = data['C'] / data['Mn']
data['Ti_Nb_Ratio'] = data['Ti'] / (data['Nb'] + 0.0001)  # 避免除零
data['C_Si_Mn_Sum'] = data['C'] + data['Si'] + data['Mn']
data['Microalloying_Sum'] = data['Ti'] + data['Nb'] + data['V'] + data['Mo']
data['Tramp_Elements'] = data['P'] + data['S'] + data['Cu']

# 2. 工艺参数组合
data['Cooling_Rate'] = (data['ActualFinalRollingTemperature'] - data['TemperatureAfterCooling']) / 10
data['Thickness_Reduction'] = data['ActualStartAfterTemperatureControl'] / data['Thickness']
data['Temperature_Drop'] = data['ActualExitTemperature'] - data['ActualFinalRollingTemperature']
data['Rolling_Intensity'] = data['Temperature_Drop'] / data['TheoreticalSlabThickness']

# 3. 碳当量 (CE)
data['Carbon_Equivalent'] = data['C'] + data['Mn']/6 + (data['Cu'] + data['Cr'])/15 + data['Mo']/10 + data['V']/5

# 分割特征和目标变量
X = data.drop(['TensileStrength'], axis=1)
y = data['TensileStrength']

# 创建训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 特征缩放
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 训练XGBoost模型
print("训练XGBoost模型...")
model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train_scaled, y_train)

# 预测
y_pred = model.predict(X_test_scaled)

# 展示5个样本的预测结果
print("\n5个样本的预测结果:")
print("样本ID | 实际值(MPa) | 预测值(MPa) | 绝对误差(MPa) | 相对误差(%)")
print("-" * 65)

# 选择5个具有代表性的样本
sample_indices = [0, len(X_test)//4, len(X_test)//2, 3*len(X_test)//4, len(X_test)-1]

for i, idx in enumerate(sample_indices):
    actual = y_test.iloc[idx]
    pred = y_pred[idx]
    abs_error = abs(actual - pred)
    rel_error = abs_error / actual * 100
    
    print(f"{i+1:^7} | {actual:^10.1f} | {pred:^10.1f} | {abs_error:^12.1f} | {rel_error:^11.1f}")

# 计算整体评估指标
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("\n整体评估指标:")
print(f"RMSE: {rmse:.2f} MPa")
print(f"MAE: {mae:.2f} MPa")
print(f"R²: {r2:.4f}")

# 计算误差分布
errors = y_test - y_pred
within_1_percent = np.sum(np.abs(errors / y_test) <= 0.01) / len(y_test) * 100
within_3_percent = np.sum(np.abs(errors / y_test) <= 0.03) / len(y_test) * 100
within_5_percent = np.sum(np.abs(errors / y_test) <= 0.05) / len(y_test) * 100

print("\n误差分布:")
print(f"误差在±1%范围内的样本比例: {within_1_percent:.1f}%")
print(f"误差在±3%范围内的样本比例: {within_3_percent:.1f}%")
print(f"误差在±5%范围内的样本比例: {within_5_percent:.1f}%") 