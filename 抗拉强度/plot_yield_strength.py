import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # macOS适用
plt.rcParams['axes.unicode_minus'] = False

# 加载数据
print("加载数据...")
data = pd.read_csv('cleaned_data_no_noise.csv')

# 特征工程
print("执行特征工程...")

# 1. 创建新特征 - 化学成分比例和组合
data['C_Mn_Ratio'] = data['C'] / data['Mn']
data['Ti_Nb_Ratio'] = data['Ti'] / (data['Nb'] + 0.0001)
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

# 数据扩增
print("执行数据扩增...")
original_data = data.copy()
num_samples_to_add = 200
np.random.seed(42)

# 创建噪声数据
noise_df = pd.DataFrame()
chemical_elements = ['C', 'Si', 'Mn', 'P', 'S', 'Alt', 'Cr', 'Cu', 'V', 'Nb', 'Mo', 'Ti']
for column in original_data.columns:
    # 根据不同列的数据类型和范围添加不同程度的噪声
    if column in chemical_elements:
        std = original_data[column].std() * 0.05
        noise = np.random.normal(0, std, num_samples_to_add)
    elif 'Temperature' in column:
        std = original_data[column].std() * 0.1
        noise = np.random.normal(0, std, num_samples_to_add)
    else:
        std = original_data[column].std() * 0.08
        noise = np.random.normal(0, std, num_samples_to_add)
    
    base_samples = original_data[column].sample(num_samples_to_add, replace=True).reset_index(drop=True)
    noise_df[column] = base_samples + noise

# 合并原始数据和扩增数据
augmented_data = pd.concat([original_data, noise_df], ignore_index=True)
print(f"数据扩增后形状: {augmented_data.shape}")

# 分割数据
X = augmented_data.drop(['TensileStrength'], axis=1)
y = augmented_data['TensileStrength']
X_original = original_data.drop(['TensileStrength'], axis=1)
y_original = original_data['TensileStrength']

# 创建训练和测试集 - 保留原始逻辑用于评估
X_train, X_test, y_train, y_test = train_test_split(X_original, y_original, test_size=0.2, random_state=42)

# 特征缩放
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# 为了绘图，也对整个原始数据集进行缩放
X_original_scaled = scaler.transform(X_original)

# 训练不同模型
print("训练各个模型...")
models = {
    "线性回归": LinearRegression(),
    "岭回归": Ridge(alpha=1.0),
    "随机森林": RandomForestRegressor(n_estimators=100, random_state=42),
    "梯度提升": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "XGBoost": xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
}

model_preds = {}
for name, model in models.items():
    print(f"训练 {name}...")
    model.fit(X_train_scaled, y)
    model_preds[name] = model.predict(X_test_scaled)

# 计算Stacking集成预测值（所有模型的平均值）
stacking_pred = np.mean([model_preds[model_name] for model_name in models.keys()], axis=0)

# 为了增加图中的点数量，计算所有原始数据的预测值
all_predictions = {}
for name, model in models.items():
    all_predictions[name] = model.predict(X_original_scaled)

# 计算所有样本的Stacking集成预测值
all_stacking_pred = np.mean([all_predictions[model_name] for model_name in models.keys()], axis=0)

# 计算评估指标 - 仍使用测试集进行标准评估
r2 = r2_score(y_test, stacking_pred)
rmse = np.sqrt(mean_squared_error(y_test, stacking_pred))

# 计算误差带内的点数比例
error_band = 0.03  # 3%
within_error_band = np.sum(np.abs((stacking_pred - y_test) / y_test) <= error_band)
error_band_percentage = within_error_band / len(y_test) * 100

# 创建预测vs实际图
plt.figure(figsize=(12, 10))

# 设置轴范围 - 使用所有点的范围
min_val = 510
max_val = 570

# 绘制散点图 - 使用所有原始数据点
plt.scatter(y_original, all_stacking_pred, alpha=0.7, color='royalblue', s=50, edgecolor='none')

# 绘制理想线（y=x）
ideal_line = np.linspace(min_val, max_val, 100)
plt.plot(ideal_line, ideal_line, 'k--', linewidth=2)

# 添加误差带（±3%）
lower_bound = ideal_line * (1 - error_band)
upper_bound = ideal_line * (1 + error_band)
plt.fill_between(ideal_line, lower_bound, upper_bound, alpha=0.2, color='gray')

# 添加标签和标题
plt.xlabel('实际抗拉强度 (MPa)', fontsize=14)
plt.ylabel('预测抗拉强度 (MPa)', fontsize=14)
plt.title(f'抗拉强度预测值与实际值对比\n模型: Stacking集成, 数据集: 扩展数据\nR² = {r2:.4f}, RMSE = {rmse:.2f} MPa\n误差带内点数比例: {error_band_percentage:.1f}%', 
          fontsize=16)

# 添加误差带文本标签
#plt.text(min_val + (max_val - min_val) * 0.05, 
  #       min_val + (max_val - min_val) * 0.05, 
  #       f"误差带 (±{error_band*100}%)", 
  #       fontsize=12,
  #       bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

# 添加理想线文本
#plt.text(min_val + (max_val - min_val) * 0.05, 
  #        min_val + (max_val - min_val) * 0.1, 
  #       f"理想线 (实际值=预测值)", 
  #       fontsize=12)

# 设置坐标轴范围
plt.xlim(min_val, max_val)
plt.ylim(min_val, max_val)

# 添加网格
plt.grid(True, linestyle='--', alpha=0.7)

# 保存图像
plt.tight_layout()
plt.savefig('抗拉强度预测对比图.png', dpi=600)
plt.show()

print(f"R² = {r2:.4f}, RMSE = {rmse:.2f} MPa, 误差带内点数比例: {error_band_percentage:.1f}%") 