import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 读取数据
print("Reading data...")
df = pd.read_csv('test1.csv')

# 计算韧性 (实测1、实测2、实测3的平均值)
df['韧性'] = df[['实测1', '实测2', '实测3']].mean(axis=1)

# 显示数据基本信息
print(f"数据集行数: {df.shape[0]}, 列数: {df.shape[1]}")

# 移除包含NaN的行
df_clean = df.dropna()
print(f"清理后数据集行数: {df_clean.shape[0]}")

# 选择特征(除去目标变量和原始测量值)
feature_columns = [col for col in df_clean.columns if col not in ['韧性', '实测1', '实测2', '实测3']]
X = df_clean[feature_columns]
y = df_clean['韧性']

# 划分训练集和测试集 (80% 训练, 20% 测试)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 标准化特征
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 训练多种模型并比较
models = {
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "Neural Network": MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
}

# 用于存储模型评估结果
results = {}

print("Training and evaluating models...")
# 训练并评估每个模型
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_scaled, y_train)
    
    # 预测
    y_pred = model.predict(X_test_scaled)
    
    # 评估
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {
        "RMSE": rmse,
        "R²": r2,
        "Predictions": y_pred
    }
    
    print(f"{name} - RMSE: {rmse:.2f}, R²: {r2:.2f}")

# 找出最佳模型
best_model_name = max(results, key=lambda k: results[k]["R²"])
print(f"\n最佳模型: {best_model_name}, R²: {results[best_model_name]['R²']:.4f}")

# 对最佳模型进行网格搜索优化
print(f"\n优化 {best_model_name} 模型的超参数...")

if best_model_name == "Random Forest":
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    best_model = RandomForestRegressor(random_state=42)
elif best_model_name == "Gradient Boosting":
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }
    best_model = GradientBoostingRegressor(random_state=42)
else:  # Neural Network
    param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate_init': [0.001, 0.01]
    }
    best_model = MLPRegressor(max_iter=1000, random_state=42)

# 执行网格搜索
grid_search = GridSearchCV(best_model, param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# 输出最佳参数
print(f"最佳参数: {grid_search.best_params_}")

# 使用优化后的模型进行预测
optimized_model = grid_search.best_estimator_
y_pred_optimized = optimized_model.predict(X_test_scaled)

# 评估优化后的模型
mse_optimized = mean_squared_error(y_test, y_pred_optimized)
rmse_optimized = np.sqrt(mse_optimized)
r2_optimized = r2_score(y_test, y_pred_optimized)

print(f"优化后的 {best_model_name} - RMSE: {rmse_optimized:.2f}, R²: {r2_optimized:.4f}")

# 可视化结果
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_optimized, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
plt.xlabel('实际韧性值')
plt.ylabel('预测韧性值')
plt.title(f'优化后的{best_model_name}模型预测效果')
plt.savefig('prediction_results.png')
plt.close()

# 特征重要性分析
if best_model_name in ["Random Forest", "Gradient Boosting"]:
    # 获取特征重要性
    importances = optimized_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # 绘制前15个重要特征
    plt.figure(figsize=(12, 8))
    plt.title('特征重要性')
    plt.bar(range(min(15, len(feature_columns))), 
            importances[indices[:15]], 
            align='center')
    plt.xticks(range(min(15, len(feature_columns))), 
              [feature_columns[i] for i in indices[:15]], 
              rotation=90)
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    
    print("\n最重要的5个特征:")
    for i in range(min(5, len(feature_columns))):
        print(f"{feature_columns[indices[i]]}: {importances[indices[i]]:.4f}")

print("\n模型训练和评估完成!") 