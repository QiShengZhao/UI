import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib as mpl
import platform
import os
from matplotlib.colors import LinearSegmentedColormap
from sklearn.inspection import permutation_importance

# 设置中文字体支持
system = platform.system()
if system == 'Darwin':  # macOS
    # 尝试多种可能的中文字体
    chinese_fonts = ['Arial Unicode MS', 'PingFang SC', 'Heiti SC', 'Microsoft YaHei', 'SimHei']
elif system == 'Windows':
    chinese_fonts = ['Microsoft YaHei', 'SimHei', 'SimSun', 'Arial Unicode MS']
else:  # Linux
    chinese_fonts = ['WenQuanYi Micro Hei', 'Noto Sans CJK JP', 'Noto Sans CJK SC', 'Droid Sans Fallback']

# 尝试设置中文字体
font_found = False
for font in chinese_fonts:
    try:
        mpl.rcParams['font.sans-serif'] = [font] + mpl.rcParams['font.sans-serif']
        mpl.rcParams['axes.unicode_minus'] = False  # 正确显示负号
        plt.rcParams['font.sans-serif'] = [font] + plt.rcParams['font.sans-serif']
        font_found = True
        print(f"Using Chinese font: {font}")
        break
    except:
        continue

if not font_found:
    print("Warning: Suitable Chinese font not found. Chinese characters might not display correctly.")

# 将中文特征名称转换为英文或代码
def translate_feature_name(name):
    translation_dict = {
        '抗拉强度': 'Tensile Strength',
        '屈服强度': 'Yield Strength',
        '终扎厚度实际值': 'Final Rolling Thickness',
        '厚度': 'Thickness',
        '水冷后温度': 'Temp After Water Cooling',
        '在炉时间(分钟)': 'Time in Furnace (min)',
        'Alt': 'Alt',
        '终扎温度实际测量值': 'Final Rolling Temp',
        '模型计算值': 'Model Calculated Value',
        '控温后开扎实际值': 'Temp After Control',
        '实际%': 'Elongation Rate (%)'
    }
    return translation_dict.get(name, name)

# 设置论文品质的图表风格
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 600,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': ':',
    'axes.axisbelow': True,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'axes.spines.top': True,
    'axes.spines.right': True,
    'axes.spines.bottom': True,
    'axes.spines.left': True,
})

# 读取数据
print("Reading data...")
df = pd.read_csv('combined_data.csv')

# 数据预处理
X = df.drop('实际%', axis=1)
y = df['实际%']

# 处理特征数据中的非数值型数据和缺失值
X = X.select_dtypes(include=[np.number])
X = X.fillna(X.mean())

# 特征和目标变量的数据拆分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 构建随机森林回归模型
print("Training Random Forest Regression model...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# 预测测试集
y_pred = rf_model.predict(X_test_scaled)

# 评估模型
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")

# 计算误差带内点数比例 (±5%)
error_band_percentage = 100 * np.mean(abs(y_test - y_pred) / y_test <= 0.05)
print(f"Percentage of points within error band (±5%): {error_band_percentage:.1f}%")

# 1. 相关性矩阵分析 - 仅使用前10个重要特征
# 获取特征重要性
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

# 获取前10个重要特征
top10_features = feature_importance.head(10)['Feature'].tolist()
top10_features.append('实际%')  # 添加目标变量

# 创建包含前10个特征和目标变量的数据框
correlation_df = df[top10_features].copy()

# 创建相关性矩阵
plt.figure(figsize=(10, 8))
corr_matrix = correlation_df.corr()

# 使用自定义的蓝红配色方案
colors = ["#053061", "#2166AC", "#4393C3", "#92C5DE", "#D1E5F0", 
          "#FFFFFF", "#FDDBC7", "#F4A582", "#D6604D", "#B2182B", "#67001F"]
cmap = LinearSegmentedColormap.from_list("custom_diverging", colors, N=256)

# 绘制热图
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True, fmt=".2f")

plt.title('重要特征相关性矩阵', fontsize=12, pad=10)
plt.tight_layout()
plt.savefig('correlation_matrix_cn.png', dpi=300, bbox_inches='tight')

# 英文版
plt.figure(figsize=(10, 8))
# 创建英文特征名称的相关性矩阵
correlation_df_en = correlation_df.copy()
correlation_df_en.columns = [translate_feature_name(col) for col in correlation_df.columns]
corr_matrix_en = correlation_df_en.corr()

sns.heatmap(corr_matrix_en, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True, fmt=".2f")

plt.title('Correlation Matrix of Important Features', fontsize=12, pad=10)
plt.tight_layout()
plt.savefig('correlation_matrix_en.png', dpi=300, bbox_inches='tight')

# 2. 残差分析
plt.figure(figsize=(8, 6))
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, alpha=0.5, color='#0072B2', edgecolor='none', s=20)
plt.axhline(y=0, color='r', linestyle='--', linewidth=1)

# 添加误差带
plt.axhline(y=1, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
plt.axhline(y=-1, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
plt.fill_between([min(y_pred), max(y_pred)], -1, 1, color='gray', alpha=0.1)

plt.xlabel('预测值')
plt.ylabel('残差 (实际值 - 预测值)')
plt.title('残差分布图')
plt.grid(True, linestyle=':', alpha=0.3)
plt.tight_layout()
plt.savefig('residual_plot_cn.png', dpi=300, bbox_inches='tight')

# 英文版
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, alpha=0.5, color='#0072B2', edgecolor='none', s=20)
plt.axhline(y=0, color='r', linestyle='--', linewidth=1)
plt.axhline(y=1, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
plt.axhline(y=-1, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
plt.fill_between([min(y_pred), max(y_pred)], -1, 1, color='gray', alpha=0.1)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals (Actual - Predicted)')
plt.title('Residual Plot')
plt.grid(True, linestyle=':', alpha=0.3)
plt.tight_layout()
plt.savefig('residual_plot_en.png', dpi=300, bbox_inches='tight')

# 3. 交叉验证
print("\nPerforming 5-fold cross-validation...")
cv_scores = cross_val_score(RandomForestRegressor(n_estimators=100, random_state=42), 
                           X_train_scaled, y_train, cv=5, scoring='r2')
print(f"Cross-validation R² scores: {cv_scores}")
print(f"Mean R²: {cv_scores.mean():.4f}, Std: {cv_scores.std():.4f}")

# 4. 混淆矩阵 (将回归问题转化为分类问题: 是否在误差带内)
y_in_error_band = abs(y_test - y_pred) / y_test <= 0.05
y_pred_in_error_band = np.ones_like(y_test, dtype=bool)  # 模型预测总是认为在误差带内

cm = confusion_matrix(y_in_error_band, y_pred_in_error_band)
tn, fp, fn, tp = cm.ravel()

# 计算指标
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"\nError Band Classification Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

plt.figure(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['误差带外', '误差带内'])
disp.plot(cmap='Blues', values_format='d')
plt.title('误差带分类混淆矩阵')
plt.tight_layout()
plt.savefig('confusion_matrix_cn.png', dpi=300, bbox_inches='tight')

# 英文版
plt.figure(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Outside Error Band', 'Within Error Band'])
disp.plot(cmap='Blues', values_format='d')
plt.title('Error Band Classification Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix_en.png', dpi=300, bbox_inches='tight')

# 5. 学习曲线
train_sizes, train_scores, test_scores = learning_curve(
    RandomForestRegressor(n_estimators=100, random_state=42), X, y, 
    train_sizes=np.linspace(0.1, 1.0, 10), cv=5, scoring='r2')

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='训练集 R²')
plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
plt.plot(train_sizes, test_mean, color='green', marker='s', markersize=5, label='验证集 R²')
plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
plt.xlabel('训练样本数量')
plt.ylabel('R² 得分')
plt.title('学习曲线')
plt.legend(loc='lower right')
plt.grid(True, linestyle=':', alpha=0.3)
plt.tight_layout()
plt.savefig('learning_curve_cn.png', dpi=300, bbox_inches='tight')

# 英文版
plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Training R²')
plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
plt.plot(train_sizes, test_mean, color='green', marker='s', markersize=5, label='Validation R²')
plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
plt.xlabel('Training Examples')
plt.ylabel('R² Score')
plt.title('Learning Curve')
plt.legend(loc='lower right')
plt.grid(True, linestyle=':', alpha=0.3)
plt.tight_layout()
plt.savefig('learning_curve_en.png', dpi=300, bbox_inches='tight')

# 6. 排列特征重要性（更稳健的特征重要性评估方法）
print("\nCalculating permutation feature importance...")
perm_importance = permutation_importance(rf_model, X_test_scaled, y_test, n_repeats=10, random_state=42)
perm_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': perm_importance.importances_mean
}).sort_values(by='Importance', ascending=False)

# 转换中文特征名称
perm_importance_df['Feature_EN'] = perm_importance_df['Feature'].apply(translate_feature_name)

print("Permutation Feature Importance (top 10):")
print(perm_importance_df.head(10))

# 绘制排列特征重要性图 - 中文版
plt.figure(figsize=(10, 6))
top_features = perm_importance_df.head(10).iloc[::-1]  # 反转以便最重要的在顶部
plt.barh(top_features['Feature'], top_features['Importance'], color='#0072B2', edgecolor='none', alpha=0.8)
plt.title('排列特征重要性 (Top 10)', fontsize=12)
plt.xlabel('重要性', fontsize=10)
plt.ylabel('特征', fontsize=10)
plt.tight_layout()
plt.grid(axis='x', linestyle=':', alpha=0.3)
plt.savefig('permutation_importance_cn.png', dpi=300, bbox_inches='tight')

# 英文版
plt.figure(figsize=(10, 6))
plt.barh(top_features['Feature_EN'], top_features['Importance'], color='#0072B2', edgecolor='none', alpha=0.8)
plt.title('Permutation Feature Importance (Top 10)', fontsize=12)
plt.xlabel('Importance', fontsize=10)
plt.ylabel('Feature', fontsize=10)
plt.tight_layout()
plt.grid(axis='x', linestyle=':', alpha=0.3)
plt.savefig('permutation_importance_en.png', dpi=300, bbox_inches='tight')

# 7. 预测分布图
plt.figure(figsize=(8, 6))
sns.histplot(y_test, color='blue', alpha=0.5, label='实际值', kde=True)
sns.histplot(y_pred, color='red', alpha=0.5, label='预测值', kde=True)
plt.xlabel('伸长率 (%)')
plt.ylabel('频数')
plt.title('实际值与预测值分布对比')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.3)
plt.tight_layout()
plt.savefig('distribution_comparison_cn.png', dpi=300, bbox_inches='tight')

# 英文版
plt.figure(figsize=(8, 6))
sns.histplot(y_test, color='blue', alpha=0.5, label='Actual Values', kde=True)
sns.histplot(y_pred, color='red', alpha=0.5, label='Predicted Values', kde=True)
plt.xlabel('Elongation Rate (%)')
plt.ylabel('Frequency')
plt.title('Distribution Comparison of Actual vs Predicted Values')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.3)
plt.tight_layout()
plt.savefig('distribution_comparison_en.png', dpi=300, bbox_inches='tight')

print("\n评估完成！所有图表已保存为中英文两个版本。") 