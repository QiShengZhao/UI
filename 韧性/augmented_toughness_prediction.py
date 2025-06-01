import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, PowerTransformer, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, VotingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import joblib

# 设置风格
plt.style.use('ggplot')
sns.set(font_scale=1.2)
sns.set_style('whitegrid')

# 读取数据
print("读取数据...")
df = pd.read_csv('test1.csv')

# 计算韧性 (实测1、实测2、实测3的平均值)
df['韧性'] = df[['实测1', '实测2', '实测3']].mean(axis=1)

# 显示数据基本信息
print(f"原始数据集行数: {df.shape[0]}, 列数: {df.shape[1]}")

# 移除包含NaN的行
df_clean = df.dropna()
print(f"清理后数据集行数: {df_clean.shape[0]}")

# 数据分析
print("\n数据分析:")
print(f"韧性值范围: {df_clean['韧性'].min():.2f} - {df_clean['韧性'].max():.2f}")
print(f"韧性值平均值: {df_clean['韧性'].mean():.2f}, 标准差: {df_clean['韧性'].std():.2f}")

# 检测异常值
z_scores = stats.zscore(df_clean['韧性'])
abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3)
df_no_outliers = df_clean[filtered_entries]
print(f"移除极端异常值后的数据行数: {df_no_outliers.shape[0]}")

# 选择特征(除去目标变量和原始测量值)
feature_columns = [col for col in df_no_outliers.columns if col not in ['韧性', '实测1', '实测2', '实测3']]
X = df_no_outliers[feature_columns]
y = df_no_outliers['韧性']

# 数据增强函数
def augment_data(X, y, augmentation_factor=3):
    """
    对数据集进行增强
    
    参数:
    - X: 特征数据
    - y: 目标变量
    - augmentation_factor: 增强倍数
    
    返回:
    - X_augmented: 增强后的特征数据
    - y_augmented: 增强后的目标变量
    """
    print(f"\n执行数据增强, 扩增倍数: {augmentation_factor}")
    X_numpy = X.values
    y_numpy = y.values
    
    # 原始数据
    X_augmented = X_numpy.copy()
    y_augmented = y_numpy.copy()
    
    n_samples, n_features = X_numpy.shape
    
    # 方法1: 添加小随机噪声
    for i in range(augmentation_factor - 1):
        # 为每个特征添加不同程度的噪声
        noise_level = 0.01 + (i * 0.005)  # 逐渐增加噪声水平
        X_noise = X_numpy + np.random.normal(0, noise_level, (n_samples, n_features))
        
        # 为目标变量添加少量噪声，保持数据分布
        y_noise = y_numpy + np.random.normal(0, 1.0, n_samples)
        
        X_augmented = np.vstack((X_augmented, X_noise))
        y_augmented = np.append(y_augmented, y_noise)
    
    # 方法2: SMOTE-like处理 (针对少数值的模拟)
    # 找出韧性值的四分位数
    q1 = np.percentile(y_numpy, 25)
    q3 = np.percentile(y_numpy, 75)
    
    # 针对极低和极高韧性值的样本进行更多的扩增
    rare_low_indices = np.where(y_numpy < q1)[0]
    rare_high_indices = np.where(y_numpy > q3)[0]
    
    # 对这些样本进行额外的增强
    for indices in [rare_low_indices, rare_high_indices]:
        if len(indices) > 0:
            for _ in range(2):  # 额外再增加两倍
                # 从稀有样本中随机选择
                selected_indices = np.random.choice(indices, size=len(indices), replace=True)
                X_selected = X_numpy[selected_indices]
                y_selected = y_numpy[selected_indices]
                
                # 添加适度的随机噪声
                X_noise = X_selected + np.random.normal(0, 0.02, X_selected.shape)
                y_noise = y_selected + np.random.normal(0, 2.0, y_selected.shape)
                
                X_augmented = np.vstack((X_augmented, X_noise))
                y_augmented = np.append(y_augmented, y_noise)
    
    print(f"原始数据样本数: {len(y_numpy)}")
    print(f"增强后数据样本数: {len(y_augmented)}")
    
    # 转换回DataFrame以保持列名称
    X_augmented_df = pd.DataFrame(X_augmented, columns=X.columns)
    
    return X_augmented_df, y_augmented

# 执行数据增强
X_augmented, y_augmented = augment_data(X, y, augmentation_factor=3)

# 划分数据集（增强后）
X_train, X_test, y_train, y_test = train_test_split(X_augmented, y_augmented, 
                                                    test_size=0.2, random_state=42)

# 原始数据集也划分一份，用于比较
X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(X, y, 
                                                                        test_size=0.2, 
                                                                        random_state=42)

# 特征缩放
print("\n执行特征缩放...")
scalers = {
    "StandardScaler": StandardScaler(),
    "RobustScaler": RobustScaler(),
    "PowerTransformer": PowerTransformer(method='yeo-johnson')
}

# 选择效果最好的缩放器
best_scaler_name = None
best_scaler_score = -np.inf

for name, scaler in scalers.items():
    X_train_scaled = scaler.fit_transform(X_train)
    
    # 使用一个简单模型快速评估
    rf = RandomForestRegressor(n_estimators=50, random_state=42)
    scores = cross_val_score(rf, X_train_scaled, y_train, cv=3, scoring='r2')
    avg_score = np.mean(scores)
    
    print(f"缩放器 {name} - 平均R²: {avg_score:.4f}")
    
    if avg_score > best_scaler_score:
        best_scaler_score = avg_score
        best_scaler_name = name

print(f"\n选择最佳缩放器: {best_scaler_name}")
scaler = scalers[best_scaler_name]

# 应用最佳缩放器
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 对原始数据集也进行相同的缩放
X_train_orig_scaled = scaler.transform(X_train_orig)
X_test_orig_scaled = scaler.transform(X_test_orig)

# 定义更多的模型进行评估
models = {
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "Neural Network": MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=2000, random_state=42),
    "AdaBoost": AdaBoostRegressor(random_state=42),
    "SVR": SVR(kernel='rbf'),
    "Ridge": Ridge(random_state=42),
    "ElasticNet": ElasticNet(random_state=42)
}

# 用于存储模型评估结果（增强数据）
results = {}
# 用于存储原始数据的结果
results_orig = {}

print("\nTraining and evaluating models with augmented data...")
# 训练并评估每个模型
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_scaled, y_train)
    
    # 在测试集上预测
    y_pred = model.predict(X_test_scaled)
    
    # 评估
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    results[name] = {
        "RMSE": rmse,
        "R²": r2,
        "MAE": mae,
        "Predictions": y_pred
    }
    
    print(f"{name} - RMSE: {rmse:.2f}, R²: {r2:.4f}, MAE: {mae:.2f}")

print("\nTraining and evaluating models with original data...")
# 使用原始数据训练和评估模型
for name, model in models.items():
    print(f"Training {name} (original data)...")
    model.fit(X_train_orig_scaled, y_train_orig)
    
    # 在测试集上预测
    y_pred_orig = model.predict(X_test_orig_scaled)
    
    # 评估
    mse_orig = mean_squared_error(y_test_orig, y_pred_orig)
    rmse_orig = np.sqrt(mse_orig)
    r2_orig = r2_score(y_test_orig, y_pred_orig)
    mae_orig = mean_absolute_error(y_test_orig, y_pred_orig)
    
    results_orig[name] = {
        "RMSE": rmse_orig,
        "R²": r2_orig,
        "MAE": mae_orig,
        "Predictions": y_pred_orig
    }
    
    print(f"{name} (original) - RMSE: {rmse_orig:.2f}, R²: {r2_orig:.4f}, MAE: {mae_orig:.2f}")
    
    # 打印对比结果
    r2_improvement = r2 - r2_orig
    print(f"R² improvement: {r2_improvement:.4f} ({r2_improvement/max(0.0001, r2_orig)*100:.1f}%)")

# 找出增强数据上的最佳模型
best_model_name = max(results, key=lambda k: results[k]["R²"])
print(f"\n增强数据的最佳模型: {best_model_name}, R²: {results[best_model_name]['R²']:.4f}")

# 对比原始数据的结果
print(f"原始数据的最佳模型: {max(results_orig, key=lambda k: results_orig[k]['R²'])}, R²: {results_orig[max(results_orig, key=lambda k: results_orig[k]['R²'])]['R²']:.4f}")

# 对最佳模型进行网格搜索优化
print(f"\n优化增强数据的 {best_model_name} 模型超参数...")

if best_model_name == "Random Forest":
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 15, 30, 45],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    best_model = RandomForestRegressor(random_state=42)
elif best_model_name == "Gradient Boosting":
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7, 9],
        'subsample': [0.8, 0.9, 1.0]
    }
    best_model = GradientBoostingRegressor(random_state=42)
elif best_model_name == "Neural Network":
    param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (150,), (50, 50), (100, 50), (100, 100)],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate_init': [0.001, 0.01, 0.1],
        'activation': ['relu', 'tanh']
    }
    best_model = MLPRegressor(max_iter=2000, random_state=42)
elif best_model_name == "AdaBoost":
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1.0],
        'loss': ['linear', 'square', 'exponential']
    }
    best_model = AdaBoostRegressor(random_state=42)
elif best_model_name == "SVR":
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.1, 0.01],
        'kernel': ['rbf', 'poly', 'sigmoid']
    }
    best_model = SVR()
elif best_model_name == "Ridge":
    param_grid = {
        'alpha': [0.1, 1.0, 10.0, 100.0],
        'solver': ['auto', 'svd', 'cholesky', 'lsqr']
    }
    best_model = Ridge(random_state=42)
else:  # ElasticNet
    param_grid = {
        'alpha': [0.1, 0.5, 1.0, 2.0],
        'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
        'max_iter': [1000, 2000, 3000]
    }
    best_model = ElasticNet(random_state=42)

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
mae_optimized = mean_absolute_error(y_test, y_pred_optimized)

print(f"优化后的 {best_model_name} - RMSE: {rmse_optimized:.2f}, R²: {r2_optimized:.4f}, MAE: {mae_optimized:.2f}")

# 创建集成模型
print("\n创建集成模型...")
# 选择表现最好的三个模型
sorted_models = sorted(results.items(), key=lambda x: x[1]['R²'], reverse=True)
top_models = [models[name] for name, _ in sorted_models[:3]]
top_model_names = [name for name, _ in sorted_models[:3]]

print(f"集成使用的模型: {', '.join(top_model_names)}")

# 创建和训练集成模型
ensemble = VotingRegressor(
    estimators=[(name, model) for name, model in zip(top_model_names, top_models)]
)
ensemble.fit(X_train_scaled, y_train)

# 预测和评估集成模型
y_pred_ensemble = ensemble.predict(X_test_scaled)
mse_ensemble = mean_squared_error(y_test, y_pred_ensemble)
rmse_ensemble = np.sqrt(mse_ensemble)
r2_ensemble = r2_score(y_test, y_pred_ensemble)
mae_ensemble = mean_absolute_error(y_test, y_pred_ensemble)

print(f"集成模型 - RMSE: {rmse_ensemble:.2f}, R²: {r2_ensemble:.4f}, MAE: {mae_ensemble:.2f}")

# 保存最佳模型
best_final_model = ensemble if r2_ensemble > r2_optimized else optimized_model
best_final_model_name = "Ensemble" if r2_ensemble > r2_optimized else best_model_name
joblib.dump(best_final_model, 'best_toughness_model.pkl')
joblib.dump(scaler, 'feature_scaler.pkl')

print(f"\n最终选择的最佳模型: {best_final_model_name}")
print("模型已保存为 'best_toughness_model.pkl'")
print("特征缩放器已保存为 'feature_scaler.pkl'")

# 可视化结果
plt.figure(figsize=(12, 8))
plt.scatter(y_test, y_pred_optimized, alpha=0.5, label='优化模型预测')
plt.scatter(y_test, y_pred_ensemble, alpha=0.5, color='red', label='集成模型预测')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
plt.xlabel('实际韧性值')
plt.ylabel('预测韧性值')
plt.title('韧性预测模型对比')
plt.legend()
plt.savefig('augmented_prediction_results.png')
plt.close()

# 特征重要性分析（对于支持的模型）
feature_importance_supported = False
if best_model_name in ["Random Forest", "Gradient Boosting", "AdaBoost"]:
    # 获取特征重要性
    importances = optimized_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    feature_importance_supported = True
    
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
    plt.savefig('augmented_feature_importance.png')
    plt.close()
    
    print("\n最重要的5个特征:")
    for i in range(min(5, len(feature_columns))):
        print(f"{feature_columns[indices[i]]}: {importances[indices[i]]:.4f}")

# 绘制原始数据与增强数据的模型效果对比图
plt.figure(figsize=(14, 8))
models_to_plot = list(results.keys())
x = np.arange(len(models_to_plot))
width = 0.35

plt.bar(x - width/2, [results[m]["R²"] for m in models_to_plot], width, label='增强数据')
plt.bar(x + width/2, [results_orig[m]["R²"] for m in models_to_plot], width, label='原始数据')

plt.ylabel('R² 分数')
plt.title('不同模型在原始数据与增强数据上的对比')
plt.xticks(x, models_to_plot, rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.savefig('data_augmentation_comparison.png')
plt.close()

print("\n数据增强和模型训练评估完成!") 