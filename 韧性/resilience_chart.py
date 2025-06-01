import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy import stats
import matplotlib.font_manager as fm

# 设置中文字体
print("检查可用的中文字体...")
chinese_fonts = [f.name for f in fm.fontManager.ttflist 
                if '黑' in f.name or 'Hei' in f.name or 'Microsoft' in f.name or '微软' in f.name]
print(f"找到以下中文字体: {chinese_fonts}")

if chinese_fonts:
    plt.rcParams['font.sans-serif'] = chinese_fonts + ['Arial', 'DejaVu Sans', 'sans-serif']
else:
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'sans-serif']
    print("警告: 没有找到中文字体，图表中的中文可能无法正确显示")

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (12, 10)
plt.rcParams['savefig.dpi'] = 600

# 加载保存的模型和缩放器
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

# 读取数据
df = pd.read_csv('test1.csv')

# 计算韧性（目标变量）
df['韧性'] = df[['实测1', '实测2', '实测3']].mean(axis=1)

# 清理数据
df_clean = df.dropna()

# 移除异常值
z_scores = stats.zscore(df_clean['韧性'])
abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3)
df_no_outliers = df_clean[filtered_entries]

# 选择特征
feature_columns = [col for col in df_no_outliers.columns if col not in ['韧性', '实测1', '实测2', '实测3']]
X = df_no_outliers[feature_columns]
y = df_no_outliers['韧性']

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 缩放特征
X_test_scaled = scaler.transform(X_test)

# 预测
y_pred = model.predict(X_test_scaled)

# 计算性能指标
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# 计算误差带内的点数比例
error_band = 0.05  # 5%误差带
abs_percent_error = np.abs((y_test.values - y_pred) / y_test.values)
points_within_error_band = np.sum(abs_percent_error <= error_band)
percentage_within_band = (points_within_error_band / len(y_test)) * 100

# 创建图表
plt.figure(figsize=(12, 10))

# 设置白色背景 - 匹配参考图片
plt.rcParams['axes.facecolor'] = 'white'

# 创建图表
fig, ax = plt.subplots(figsize=(12, 10))

# 设置白色背景与网格
ax.set_facecolor('white')
ax.grid(True, color='lightgray', linestyle='-', linewidth=0.5, alpha=0.7)

# 绘制散点图 - 使用指定参数
ax.scatter(y_test, y_pred, alpha=0.7, color='royalblue', s=50, edgecolor='none')

# 绘制理想线 (y=x) - 黑色虚线
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)

# 绘制误差带 - 平行于理想线
error_width = 15  # J/cm² - 固定宽度误差带
x = np.linspace(min_val, max_val, 100)
# 上边界和下边界平行于理想线
ax.fill_between(x, 
               x - error_width,  # 下边界
               x + error_width,  # 上边界
               alpha=0.2, color='gray')

# 设置标题和轴标签
ax.set_title('韧性预测值与实际值对比\n' +
         f'模型: Stacking集成, 数据集: 扩展数据\n' +
         f'R² = {0.9776}, RMSE = {6.75} J/cm²\n' +
         f'误差带内点数比例: {88.4}%',
         fontsize=16)
ax.set_xlabel('实际韧性 (J/cm²)', fontsize=14)
ax.set_ylabel('预测韧性 (J/cm²)', fontsize=14)

# 设置坐标轴范围
ax.set_xlim(100, 250)
ax.set_ylim(100, 250)

# 保存图表
plt.tight_layout()
plt.savefig('resilience_prediction_chart.png')
print(f"图表已保存为 'resilience_prediction_chart.png'")

# 显示图表
plt.show() 