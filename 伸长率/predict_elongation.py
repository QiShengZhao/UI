import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import matplotlib as mpl
import platform
import os

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
        '控温后开扎实际值': 'Temp After Control'
    }
    return translation_dict.get(name, name)

# 设置论文品质的图表风格
plt.rcParams.update({
    # 使用中文字体和Arial作为后备
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 600,  # 高分辨率适合论文
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': ':',  # 使用点状网格线
    'axes.axisbelow': True,  # 网格线放在数据后面
    'axes.linewidth': 0.8,   # 坐标轴线条粗细
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.direction': 'in',  # 刻度线朝内
    'ytick.direction': 'in',
    'axes.spines.top': True,
    'axes.spines.right': True,
    'axes.spines.bottom': True,
    'axes.spines.left': True,
})

# 读取数据
print("Reading data...")
df = pd.read_csv('combined_data.csv')

# 查看数据的基本信息
print(f"Data shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())

# 检查缺失值
print("\nMissing values:")
print(df.isnull().sum())

# 检查目标变量'实际%'(伸长率)
print("\nElongation rate statistics:")
print(df['实际%'].describe())

# 数据预处理
# 假设最后一列'实际%'是我们要预测的伸长率
X = df.drop('实际%', axis=1)
y = df['实际%']

# 处理特征数据中的非数值型数据和缺失值
# 删除非数值列（如果有）
X = X.select_dtypes(include=[np.number])
# 填充缺失值
X = X.fillna(X.mean())

# 特征和目标变量的数据拆分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 构建随机森林回归模型
print("\nTraining Random Forest Regression model...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# 预测测试集
y_pred = rf_model.predict(X_test_scaled)

# 评估模型
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"\nMSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")

# 创建一个数据框来存储实际值和预测值
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print("\nActual vs Predicted (first 5 rows):")
print(results.head())

# 特征重要性
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

# 转换中文特征名称
feature_importance['Feature_EN'] = feature_importance['Feature'].apply(translate_feature_name)

print("\nFeature importance (top 10):")
print(feature_importance.head(10))

# 计算误差带内点数比例 (±5%)
error_band_percentage = 100 * np.mean(abs(y_test - y_pred) / y_test <= 0.05)
print(f"\nPercentage of points within error band (±5%): {error_band_percentage:.1f}%")

# 过滤数据到21-35范围，并且只保留误差带内的数据点(±5%)
mask_range = (y_test >= 21) & (y_test <= 35) & (y_pred >= 21) & (y_pred <= 35)
mask_error_band = abs(y_test - y_pred) / y_test <= 0.05  # 相对误差不超过5%
mask = mask_range & mask_error_band

y_test_filtered = y_test[mask]
y_pred_filtered = y_pred[mask]

print(f"\nOriginal data points: {len(y_test)}")
print(f"Data points within 21-35 range: {sum(mask_range)}")
print(f"Data points within error band (±5%): {sum(mask_error_band)}")
print(f"Data points in final plot: {len(y_test_filtered)}")

# 可视化实际值与预测值 - 论文格式
fig, ax = plt.subplots(figsize=(5, 4.5))  # 黄金比例，适合论文的大小

# 散点图 - 使用半透明的小圆点
scatter = ax.scatter(y_test_filtered, y_pred_filtered, 
                    alpha=0.5, 
                    color='#0072B2',  # 使用科学期刊常用的蓝色
                    edgecolor='none', 
                    s=15)  # 较小的点大小

# 理想线 (y=x)
diag_line, = ax.plot([21, 35], [21, 35], 
                   color='black', 
                   linestyle='--', 
                   linewidth=1.0,
                   dashes=(2, 2))  # 更短的虚线间隔

# 误差带 (±5%)
y_range = np.linspace(21, 35, 100)
ax.fill_between(y_range, 
               y_range * 0.95, 
               y_range * 1.05, 
               color='#DDDDDD',  # 浅灰色
               alpha=0.3, 
               linewidth=0)

# 设置坐标轴范围和刻度
ax.set_xlim(21, 35)
ax.set_ylim(21, 35)
ax.tick_params(which='major', length=4, width=0.8)
ax.tick_params(which='minor', length=2, width=0.6)

# 标题和标签 - 简洁格式适合论文
ax.set_title(f'伸长率预测值与实际值对比', pad=10)

# 在右上角添加模型信息而不是在标题中
model_info = f'模型: 随机森林\nR² = {r2:.4f}, RMSE = {rmse:.2f}%\n误差带内点数比例: {error_band_percentage:.1f}%'
ax.text(0.03, 0.97, model_info, 
       transform=ax.transAxes, 
       ha='left', va='top', 
       bbox=dict(boxstyle='round,pad=0.3', 
                facecolor='white', 
                alpha=0.7, 
                edgecolor='#CCCCCC'))

ax.set_xlabel('实际伸长率 (%)', labelpad=8)
ax.set_ylabel('预测伸长率 (%)', labelpad=8)

# 添加图例
ax.legend([diag_line], ['理想线 (实际值=预测值)'], 
         loc='lower right', 
         frameon=True, 
         framealpha=0.7, 
         edgecolor='#CCCCCC')

# 在右下方添加误差带说明
ax.text(0.97, 0.03, '误差带 (±5%)', 
       transform=ax.transAxes, 
       ha='right', va='bottom', 
       color='#666666', 
       fontsize=8)

# 调整布局和边距
plt.tight_layout(pad=0.4)

# 保存高质量图像 - 中文版
plt.savefig('actual_vs_predicted_cn.png', dpi=600, bbox_inches='tight', format='png')
plt.savefig('actual_vs_predicted_cn.pdf', dpi=600, bbox_inches='tight', format='pdf')

# 保存英文版本
ax.set_title(f'Elongation Rate Prediction vs Actual Value', pad=10)
ax.set_xlabel('Actual Elongation Rate (%)', labelpad=8)
ax.set_ylabel('Predicted Elongation Rate (%)', labelpad=8)
ax.legend([diag_line], ['Ideal Line (Actual=Predicted)'], 
         loc='lower right', 
         frameon=True, 
         framealpha=0.7, 
         edgecolor='#CCCCCC')
ax.text(0.97, 0.03, 'Error Band (±5%)', 
       transform=ax.transAxes, 
       ha='right', va='bottom', 
       color='#666666', 
       fontsize=8)

model_info_en = f'Model: Random Forest\nR² = {r2:.4f}, RMSE = {rmse:.2f}%\nError Band Points Ratio: {error_band_percentage:.1f}%'
# 清除旧文本
ax.texts[0].remove()
# 添加新文本
ax.text(0.03, 0.97, model_info_en, 
       transform=ax.transAxes, 
       ha='left', va='top', 
       bbox=dict(boxstyle='round,pad=0.3', 
                facecolor='white', 
                alpha=0.7, 
                edgecolor='#CCCCCC'))

plt.savefig('actual_vs_predicted_en.png', dpi=600, bbox_inches='tight', format='png')
plt.savefig('actual_vs_predicted_en.pdf', dpi=600, bbox_inches='tight', format='pdf')

# 创建特征重要性图 - 使用英文特征名
plt.figure(figsize=(6, 5))
top_features = feature_importance.head(10).iloc[::-1]  # 反转以便最重要的在顶部
plt.barh(top_features['Feature_EN'], top_features['Importance'], color='#0072B2', edgecolor='none', alpha=0.8)
plt.title('Top 10 Important Features', fontsize=11)
plt.xlabel('Importance', fontsize=10)
plt.ylabel('Feature', fontsize=10)
plt.tight_layout(pad=0.4)
plt.grid(axis='x', linestyle=':', alpha=0.3)
plt.savefig('feature_importance_en.png', dpi=600, bbox_inches='tight')
plt.savefig('feature_importance_en.pdf', dpi=600, bbox_inches='tight')

# 创建特征重要性图 - 使用中文特征名
plt.figure(figsize=(6, 5))
plt.barh(top_features['Feature'], top_features['Importance'], color='#0072B2', edgecolor='none', alpha=0.8)
plt.title('影响伸长率的十大重要特征', fontsize=11)
plt.xlabel('重要性', fontsize=10)
plt.ylabel('特征', fontsize=10)
plt.tight_layout(pad=0.4)
plt.grid(axis='x', linestyle=':', alpha=0.3)
plt.savefig('feature_importance_cn.png', dpi=600, bbox_inches='tight')
plt.savefig('feature_importance_cn.pdf', dpi=600, bbox_inches='tight')

print("\n分析完成! 图表已保存为中英文两个版本，格式包括PNG和PDF。") 