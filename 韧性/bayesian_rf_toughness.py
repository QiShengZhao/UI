import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import joblib

# 尝试导入scikit-optimize，如果没有安装则提示安装
try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer, Categorical
except ImportError:
    print("需要安装scikit-optimize库进行贝叶斯优化")
    print("请运行: pip install scikit-optimize")
    exit(1)

# 设置matplotlib参数，以英文显示避免中文问题
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'sans-serif']
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['savefig.dpi'] = 300

print("开始加载和处理数据...")
# 读取数据
df = pd.read_csv('test1.csv')

# 计算韧性 (实测1、实测2、实测3的平均值)
df['韧性'] = df[['实测1', '实测2', '实测3']].mean(axis=1)

# 显示数据基本信息
print(f"原始数据集行数: {df.shape[0]}, 列数: {df.shape[1]}")

# 移除包含NaN的行
df_clean = df.dropna()
print(f"清理后数据集行数: {df_clean.shape[0]}")

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

# 特征缩放
print("\n执行特征缩放...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 使用贝叶斯优化随机森林模型
print("\n开始贝叶斯优化随机森林模型参数...")

# 定义搜索空间
search_spaces = {
    'n_estimators': Integer(50, 300),
    'max_depth': Integer(5, 50),
    'min_samples_split': Integer(2, 20),
    'min_samples_leaf': Integer(1, 10),
    'max_features': Categorical(['sqrt', 'log2', None]),
    'bootstrap': Categorical([True, False])
}

# 创建基础随机森林模型
rf_model = RandomForestRegressor(random_state=42)

# 创建贝叶斯搜索对象
bayes_search = BayesSearchCV(
    rf_model,
    search_spaces,
    n_iter=50,  # 贝叶斯搜索迭代次数
    cv=5,       # 交叉验证折数
    n_jobs=-1,  # 使用所有可用CPU
    scoring='r2',
    random_state=42,
    verbose=1
)

# 执行搜索
print("开始执行贝叶斯搜索，这可能需要一些时间...")
bayes_search.fit(X_train_scaled, y_train)

# 输出最佳参数
print(f"\n最佳参数: {bayes_search.best_params_}")
print(f"最佳交叉验证得分: {bayes_search.best_score_:.4f}")

# 获取最佳模型
best_rf_model = bayes_search.best_estimator_

# 在测试集上评估最佳模型
y_pred = best_rf_model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"\n测试集上的模型性能:")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.4f}")
print(f"MAE: {mae:.2f}")

# 保存模型和缩放器
joblib.dump(best_rf_model, 'best_rf_model.pkl')
joblib.dump(scaler, 'rf_feature_scaler.pkl')
print("\n模型和缩放器已保存.")

# 可视化
print("\n创建可视化...")

# 创建预测值与实际值的散点图
plt.figure(figsize=(10, 8))
plt.scatter(y_test, y_pred, alpha=0.6, edgecolor='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.title('Random Forest with Bayesian Optimization: Actual vs Predicted')
plt.xlabel('Actual Toughness (J/cm²)')
plt.ylabel('Predicted Toughness (J/cm²)')

# 添加统计信息
stats_text = f'$R^2 = {r2:.4f}$\nRMSE = {rmse:.2f}\nMAE = {mae:.2f}'
plt.annotate(stats_text, xy=(0.05, 0.95), xycoords='axes fraction',
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
             ha='left', va='top')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('rf_bayesian_scatter.png')
plt.close()

# 创建残差图
plt.figure(figsize=(10, 8))
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, alpha=0.6, edgecolor='k')
plt.axhline(y=0, color='r', linestyle='-', lw=2)
plt.title('Random Forest with Bayesian Optimization: Residuals')
plt.xlabel('Predicted Toughness (J/cm²)')
plt.ylabel('Residuals (Actual - Predicted)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('rf_bayesian_residuals.png')
plt.close()

# 特征重要性分析
importances = best_rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 10))
plt.title('Feature Importance')
plt.barh(range(min(15, len(feature_columns))), 
         importances[indices[:15]], 
         align='center')
plt.yticks(range(min(15, len(feature_columns))), 
           [feature_columns[i] for i in indices[:15]])
plt.xlabel('Relative Importance')
plt.gca().invert_yaxis()  # 让最重要的特征显示在顶部
plt.tight_layout()
plt.savefig('rf_bayesian_feature_importance.png')
plt.close()

print("\n最重要的5个特征:")
for i in range(min(5, len(feature_columns))):
    print(f"{feature_columns[indices[i]]}: {importances[indices[i]]:.4f}")

# 创建学习曲线
results = pd.DataFrame(bayes_search.cv_results_)
best_index = bayes_search.best_index_

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(results) + 1), -results['mean_test_score'], 'o-', label='Mean Test Score')
plt.axvline(x=best_index + 1, color='r', linestyle='--', label=f'Best Iteration ({best_index + 1})')
plt.title('Bayesian Optimization Learning Curve')
plt.xlabel('Iteration')
plt.ylabel('Negative Mean R² Score')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('rf_bayesian_learning_curve.png')
plt.close()

# 比较原始随机森林与贝叶斯优化后的随机森林
print("\n比较原始随机森林与贝叶斯优化后的随机森林...")
# 训练原始随机森林模型
original_rf = RandomForestRegressor(n_estimators=100, random_state=42)
original_rf.fit(X_train_scaled, y_train)
y_pred_orig = original_rf.predict(X_test_scaled)
r2_orig = r2_score(y_test, y_pred_orig)
rmse_orig = np.sqrt(mean_squared_error(y_test, y_pred_orig))
mae_orig = mean_absolute_error(y_test, y_pred_orig)

print(f"\n原始随机森林 - RMSE: {rmse_orig:.2f}, R²: {r2_orig:.4f}, MAE: {mae_orig:.2f}")
print(f"贝叶斯优化随机森林 - RMSE: {rmse:.2f}, R²: {r2:.4f}, MAE: {mae:.2f}")
print(f"R²提升: {r2 - r2_orig:.4f} ({(r2 - r2_orig) / r2_orig * 100:.2f}%)")
print(f"RMSE降低: {rmse_orig - rmse:.2f} ({(rmse_orig - rmse) / rmse_orig * 100:.2f}%)")

print("\n所有处理完成!") 