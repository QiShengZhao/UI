import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # macOS适用
plt.rcParams['axes.unicode_minus'] = False

# 创建绘制标准对比图的函数
def plot_prediction_comparison(y_true, y_pred, model_name, dataset_type="原始数据"):
    """
    绘制预测值与实际值的对比图，包含误差带和统计信息
    
    参数:
    y_true: 实际值
    y_pred: 预测值
    model_name: 模型名称
    dataset_type: 数据集类型
    """
    # 计算评估指标
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # 计算误差带内的点数比例 (±5%)
    error_band = 0.05
    within_error_band = np.sum(np.abs((y_pred - y_true) / y_true) <= error_band)
    error_band_percentage = within_error_band / len(y_true) * 100
    
    # 创建图形
    plt.figure(figsize=(12, 10))
    
    # 散点图
    plt.scatter(y_true, y_pred, alpha=0.7, color='royalblue', s=50, edgecolor='none')
    
    # 理想线 (y=x)
    min_val = min(y_true.min(), y_pred.min()) * 0.95
    max_val = max(y_true.max(), y_pred.max()) * 1.05
    ideal_line = np.linspace(min_val, max_val, 100)
    plt.plot(ideal_line, ideal_line, 'k--', linewidth=2)
    
    # 误差带 (±5%)
    lower_bound = ideal_line * (1 - error_band)
    upper_bound = ideal_line * (1 + error_band)
    plt.fill_between(ideal_line, lower_bound, upper_bound, alpha=0.2, color='gray')
    
    # 在误差带中添加文本标签
    plt.text(min_val + (max_val - min_val) * 0.15, 
             min_val + (max_val - min_val) * 0.1, 
             f"误差带 (±{error_band*100}%)", 
             fontsize=12, 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
    
    # 添加理想线的文本说明
    plt.text(min_val + (max_val - min_val) * 0.15, 
             min_val + (max_val - min_val) * 0.15, 
             f"理想线 (实际值=预测值)", 
             fontsize=12)
    
    # 设置坐标轴范围
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    
    # 添加标签和标题
    plt.xlabel('实际抗拉强度 (MPa)', fontsize=14)
    plt.ylabel('预测抗拉强度 (MPa)', fontsize=14)
    plt.title(f'抗拉强度预测值与实际值对比\n模型: {model_name}, 数据集: {dataset_type}\nR² = {r2:.4f}, RMSE = {rmse:.2f} MPa\n误差带内点数比例: {error_band_percentage:.1f}%', 
              fontsize=16)
    
    # 添加网格线
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(f'prediction_comparison_{dataset_type}.png', dpi=300)
    plt.close()
    
    return r2, rmse, error_band_percentage

# 加载数据
print("加载数据...")
data = pd.read_csv('cleaned_data_no_noise.csv')

# 探索性数据分析
print("数据形状:", data.shape)
print("\n数据列名:")
print(data.columns.tolist())
print("\n数据摘要:")
print(data.describe())

# 检查缺失值
print("\n缺失值:")
print(data.isnull().sum())

# 分析TensileStrength的相关性
correlations = data.corr()['TensileStrength'].sort_values(ascending=False)
print("\nTensileStrength相关性:")
print(correlations)

# 可视化相关矩阵
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=False, cmap='coolwarm')
plt.title('相关矩阵')
plt.tight_layout()
plt.savefig('correlation_matrix.png')

# 绘制与TensileStrength相关性最高的5个特征
top_features = correlations.index[1:6]  # 排除TensileStrength本身
plt.figure(figsize=(15, 10))
for i, feature in enumerate(top_features):
    plt.subplot(2, 3, i+1)
    sns.scatterplot(x=feature, y='TensileStrength', data=data)
    plt.title(f'TensileStrength vs {feature}')
plt.tight_layout()
plt.savefig('top_correlations.png')

# 化学成分分析
chemical_elements = ['C', 'Si', 'Mn', 'P', 'S', 'Alt', 'Cr', 'Cu', 'V', 'Nb', 'Mo', 'Ti']
plt.figure(figsize=(15, 10))
for i, element in enumerate(chemical_elements):
    plt.subplot(3, 4, i+1)
    sns.scatterplot(x=element, y='TensileStrength', data=data)
    plt.title(f'TensileStrength vs {element}')
plt.tight_layout()
plt.savefig('chemical_composition.png')

# 特征工程
print("\n执行特征工程...")

# 1. 创建新特征 - 化学成分比例和组合
data['C_Mn_Ratio'] = data['C'] / data['Mn']
data['Ti_Nb_Ratio'] = data['Ti'] / (data['Nb'] + 0.0001)  # 避免除零
data['C_Si_Mn_Sum'] = data['C'] + data['Si'] + data['Mn']
data['Microalloying_Sum'] = data['Ti'] + data['Nb'] + data['V'] + data['Mo']
data['Tramp_Elements'] = data['P'] + data['S'] + data['Cu']

# 2. 工艺参数组合
data['Cooling_Rate'] = (data['ActualFinalRollingTemperature'] - data['TemperatureAfterCooling']) / 10  # 假设冷却时间为常数
data['Thickness_Reduction'] = data['ActualStartAfterTemperatureControl'] / data['Thickness']
data['Temperature_Drop'] = data['ActualExitTemperature'] - data['ActualFinalRollingTemperature']
data['Rolling_Intensity'] = data['Temperature_Drop'] / data['TheoreticalSlabThickness']

# 3. 碳当量 (CE) - 钢铁冶金学中常用的综合成分指标
data['Carbon_Equivalent'] = data['C'] + data['Mn']/6 + (data['Cu'] + data['Cr'])/15 + data['Mo']/10 + data['V']/5

# 数据扩增
print("\n执行数据扩增...")
original_data = data.copy()
num_samples_to_add = 200
np.random.seed(42)

# 创建噪声数据
noise_df = pd.DataFrame()
for column in original_data.columns:
    # 根据不同列的数据类型和范围添加不同程度的噪声
    if column in chemical_elements:
        # 化学成分添加较小的噪声
        std = original_data[column].std() * 0.05
        noise = np.random.normal(0, std, num_samples_to_add)
    elif 'Temperature' in column:
        # 温度添加中等噪声
        std = original_data[column].std() * 0.1
        noise = np.random.normal(0, std, num_samples_to_add)
    else:
        # 其他特征添加一般噪声
        std = original_data[column].std() * 0.08
        noise = np.random.normal(0, std, num_samples_to_add)
    
    # 随机选择基础样本，添加噪声生成新样本
    base_samples = original_data[column].sample(num_samples_to_add, replace=True).reset_index(drop=True)
    noise_df[column] = base_samples + noise

# 合并原始数据和扩增数据
augmented_data = pd.concat([original_data, noise_df], ignore_index=True)
print(f"数据扩增后形状: {augmented_data.shape}")

# 分割特征和目标变量
X = augmented_data.drop(['TensileStrength'], axis=1)
y = augmented_data['TensileStrength']

# 创建原始数据集的测试集(保持不变)，用于最终评估
X_original = original_data.drop(['TensileStrength'], axis=1)
y_original = original_data['TensileStrength']
X_train, X_test, y_train, y_test = train_test_split(X_original, y_original, test_size=0.2, random_state=42)

# 使用全部扩增数据集训练模型
X_train_aug = X
y_train_aug = y

# 特征缩放
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_aug)
X_test_scaled = scaler.transform(X_test)

# 保存原始列名
feature_names = X.columns

# 多项式特征 (只用于线性模型)
poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

# 交叉验证
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# 初始化模型
models = {
    "线性回归": LinearRegression(),
    "岭回归": Ridge(alpha=1.0),
    "随机森林": RandomForestRegressor(n_estimators=100, random_state=42),
    "梯度提升": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "XGBoost": xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
}

# 模型评估结果存储
results = {}
feature_importance_data = {}

print("\n模型训练和评估:")
all_models_preds = {}

for name, model in models.items():
    print(f"\n训练 {name}...")
    
    # 根据模型类型选择合适的特征集
    if name in ["线性回归", "岭回归"]:
        X_train_model = X_train_poly
        X_test_model = X_test_poly
    else:
        X_train_model = X_train_scaled
        X_test_model = X_test_scaled
    
    # 训练模型
    model.fit(X_train_model, y_train_aug)
    
    # 预测
    y_pred = model.predict(X_test_model)
    all_models_preds[name] = y_pred
    
    # 评估指标
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    results[name] = {
        "R²": r2,
        "RMSE": rmse,
        "MAE": mae
    }
    
    # 如果模型支持特征重要性
    if hasattr(model, 'feature_importances_'):
        if name in ["线性回归", "岭回归"]:
            # 对于多项式特征，需要特殊处理
            pass
        else:
            importance = model.feature_importances_
            feature_importance_data[name] = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance
            }).sort_values('Importance', ascending=False)
    
    # 绘制实际vs预测图（使用新的标准对比图函数）
    plot_prediction_comparison(y_test, y_pred, name, f"{name}_扩增数据")

# 输出模型评估结果
print("\n模型评估结果汇总:")
results_df = pd.DataFrame(results).T
print(results_df)

# 选择最佳模型
best_model_name = results_df['R²'].idxmax()
best_r2 = results_df.loc[best_model_name, 'R²']
print(f"\n最佳模型: {best_model_name}, R² 得分: {best_r2:.4f}")

# 使用Stacking集成方法
print("\n创建Stacking集成模型...")
# 计算所有基础模型预测值的平均值
stacking_pred = np.mean([all_models_preds[model_name] for model_name in all_models_preds], axis=0)
# 绘制Stacking模型的实际vs预测图
stacking_r2, stacking_rmse, stacking_error_band = plot_prediction_comparison(
    y_test, stacking_pred, "Stacking集成", "扩展数据"
)
print(f"Stacking集成模型 - R² = {stacking_r2:.4f}, RMSE = {stacking_rmse:.2f}, 误差带内点数比例: {stacking_error_band:.1f}%")

# 显示最佳模型的特征重要性
if best_model_name in feature_importance_data:
    print(f"\n{best_model_name}的前15个重要特征:")
    print(feature_importance_data[best_model_name].head(15))
    
    # 可视化特征重要性
    plt.figure(figsize=(12, 8))
    top_features = feature_importance_data[best_model_name].head(15)
    sns.barplot(x='Importance', y='Feature', data=top_features)
    plt.title(f'{best_model_name} - 特征重要性')
    plt.tight_layout()
    plt.savefig(f'{best_model_name}_feature_importance.png')

# 分析工程特征的效果
print("\n工程特征的相关性:")
engineered_features = ['C_Mn_Ratio', 'Ti_Nb_Ratio', 'C_Si_Mn_Sum', 
                      'Microalloying_Sum', 'Tramp_Elements', 'Cooling_Rate', 
                      'Thickness_Reduction', 'Temperature_Drop', 
                      'Rolling_Intensity', 'Carbon_Equivalent']

eng_correlations = augmented_data[engineered_features + ['TensileStrength']].corr()['TensileStrength'].sort_values(ascending=False)
print(eng_correlations)

# 可视化工程特征的重要性
plt.figure(figsize=(12, 8))
top_eng_features = pd.DataFrame({
    'Feature': engineered_features,
    'Correlation': [abs(eng_correlations[f]) for f in engineered_features]
}).sort_values('Correlation', ascending=False)

sns.barplot(x='Correlation', y='Feature', data=top_eng_features)
plt.title('工程特征化相关性重要度')
plt.tight_layout()
plt.savefig('engineered_feature_importance.png')

print("\n分析完成。请查看生成的图表获取可视化分析。")
