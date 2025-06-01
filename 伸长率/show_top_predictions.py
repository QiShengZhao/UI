import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import matplotlib as mpl
from tabulate import tabulate

# 设置中文字体支持
try:
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Microsoft YaHei', 'SimHei'] + plt.rcParams['font.sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    print("中文字体设置成功")
except:
    print("中文字体设置失败，可能会显示为方框")

# 读取数据
print("读取数据...")
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
print("训练随机森林回归模型...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# 预测测试集
y_pred = rf_model.predict(X_test_scaled)

# 特征重要性
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\n特征重要性 (Top 10):")
print(feature_importance.head(10))

# 获取Top 10重要特征
top10_features = feature_importance.head(10)['Feature'].tolist()

# 创建一个包含Top 10特征、实际值和预测值的数据框
results_df = pd.DataFrame({
    '实际值': y_test,
    '预测值': y_pred,
    '绝对误差': np.abs(y_test - y_pred),
    '相对误差(%)': np.abs(y_test - y_pred) / y_test * 100
})

# 将Top 10特征的值添加到结果数据框中
for feature in top10_features:
    results_df[feature] = X_test[feature].values

# 过滤在21-35范围内的数据点
mask_range = (results_df['实际值'] >= 21) & (results_df['实际值'] <= 35) & (results_df['预测值'] >= 21) & (results_df['预测值'] <= 35)
filtered_results = results_df[mask_range]

# 选择8组有代表性的数据点
# 1. 预测最准确的2个样本
most_accurate = filtered_results.sort_values(by='绝对误差').head(2)
# 2. 预测误差中等的4个样本
medium_error = filtered_results[(filtered_results['绝对误差'] >= filtered_results['绝对误差'].median() - 0.1) & 
                              (filtered_results['绝对误差'] <= filtered_results['绝对误差'].median() + 0.1)].head(4)
# 3. 预测误差较大但仍在误差带内的2个样本
larger_error = filtered_results[(filtered_results['相对误差(%)'] < 5) & 
                              (filtered_results['绝对误差'] > filtered_results['绝对误差'].quantile(0.9))].head(2)

# 如果larger_error不足2个，从相对误差最大的样本中选择
if len(larger_error) < 2:
    additional_samples = filtered_results.sort_values(by='相对误差(%)', ascending=False).head(2 - len(larger_error))
    larger_error = pd.concat([larger_error, additional_samples])

# 合并所有选择的样本
selected_samples = pd.concat([most_accurate, medium_error, larger_error])

# 如果不足8个样本，随机选择剩余的
if len(selected_samples) < 8:
    remaining = filtered_results[~filtered_results.index.isin(selected_samples.index)]
    additional = remaining.sample(n=min(8 - len(selected_samples), len(remaining)), random_state=42)
    selected_samples = pd.concat([selected_samples, additional])

# 确保只有8个样本
selected_samples = selected_samples.head(8)

# 重置索引
selected_samples = selected_samples.reset_index(drop=True)

# 创建一个更好的显示格式
display_df = selected_samples.copy()
display_df.index = display_df.index + 1  # 从1开始的索引

# 格式化显示的数据
formatted_df = pd.DataFrame({
    '样本': display_df.index,
    '实际值': display_df['实际值'].round(2),
    '预测值': display_df['预测值'].round(2),
    '绝对误差': display_df['绝对误差'].round(3),
    '相对误差(%)': display_df['相对误差(%)'].round(2)
})

# 为每个Top 10特征添加列 - 只选择前5个最重要的特征，避免表格过宽
for feature in top10_features[:5]:  # 只显示前5个最重要特征
    # 根据特征名称选择合适的小数位数
    if feature in ['抗拉强度', '屈服强度', '终扎厚度实际值', '厚度', '水冷后温度']:
        formatted_df[feature] = display_df[feature].round(1)
    else:
        formatted_df[feature] = display_df[feature].round(3)

# 使用tabulate打印美观的表格
print("\n基于Top 5重要特征的8组预测样本:")
print(tabulate(formatted_df, headers='keys', tablefmt='pretty', showindex=False))

# 保存为CSV文件
formatted_df.to_csv('top_features_predictions.csv', index=False, encoding='utf-8-sig')
print("\n数据已保存到 top_features_predictions.csv")

# 直接使用HTML格式保存表格
html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>伸长率预测结果</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
        th, td {{ padding: 8px; text-align: left; border: 1px solid #ddd; }}
        th {{ background-color: #f2f2f2; color: #333; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        .good {{ background-color: #c6efce; }}
        .medium {{ background-color: #ffeb9c; }}
        .bad {{ background-color: #ffc7ce; }}
        caption {{ font-weight: bold; font-size: 1.2em; margin-bottom: 10px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1, h2 {{ color: #333; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>伸长率预测分析</h1>
        <h2>基于随机森林模型的预测结果</h2>
        
        <table>
            <caption>基于Top 5重要特征的8组预测样本</caption>
            <thead>
                <tr>
                    <th>样本</th>
                    <th>实际值</th>
                    <th>预测值</th>
                    <th>绝对误差</th>
                    <th>相对误差(%)</th>
                    <th>抗拉强度</th>
                    <th>屈服强度</th>
                    <th>终扎厚度实际值</th>
                    <th>厚度</th>
                    <th>水冷后温度</th>
                </tr>
            </thead>
            <tbody>
"""

# 添加表格数据行
for _, row in formatted_df.iterrows():
    # 根据误差大小确定CSS类
    if row['绝对误差'] < 0.1:
        error_class = "good"
    elif row['绝对误差'] < 0.5:
        error_class = "medium"
    else:
        error_class = "bad"
    
    html_content += f"""
                <tr>
                    <td>{row['样本']}</td>
                    <td>{row['实际值']:.2f}</td>
                    <td>{row['预测值']:.2f}</td>
                    <td class="{error_class}">{row['绝对误差']:.3f}</td>
                    <td class="{error_class}">{row['相对误差(%)']:.2f}%</td>
                    <td>{row['抗拉强度']:.1f}</td>
                    <td>{row['屈服强度']:.1f}</td>
                    <td>{row['终扎厚度实际值']:.1f}</td>
                    <td>{row['厚度']:.1f}</td>
                    <td>{row['水冷后温度']:.1f}</td>
                </tr>"""

# 完成HTML文档
html_content += """
            </tbody>
        </table>
        <p>注：颜色标记表示误差大小：绿色 - 误差小，黄色 - 误差中等，红色 - 误差较大</p>
    </div>
</body>
</html>
"""

# 保存HTML文件
with open('top_features_predictions.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

print("详细的HTML表格已保存到 top_features_predictions.html")

# 可视化Top 5特征与预测结果的关系
plt.figure(figsize=(15, 10))
for i, feature in enumerate(top10_features[:5]):  # 只显示Top 5特征
    plt.subplot(2, 3, i+1)
    plt.scatter(X_test[feature], y_test, alpha=0.5, label='实际值', color='blue', s=20)
    plt.scatter(X_test[feature], y_pred, alpha=0.5, label='预测值', color='red', s=20)
    plt.xlabel(feature)
    plt.ylabel('伸长率 (%)')
    plt.title(f'{feature} 与伸长率关系')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig('top_features_relationships.png', dpi=300, bbox_inches='tight')
print("特征关系图已保存到 top_features_relationships.png")

# 创建一个专门展示8个样本点的表格图
plt.figure(figsize=(10, 6))
x = np.arange(len(formatted_df))
width = 0.35

plt.bar(x - width/2, formatted_df['实际值'], width, label='实际值', color='#4878D0')
plt.bar(x + width/2, formatted_df['预测值'], width, label='预测值', color='#EE854A')

plt.xlabel('样本编号')
plt.ylabel('伸长率 (%)')
plt.title('8组样本的实际值与预测值对比')
plt.xticks(x, formatted_df['样本'])
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6, axis='y')

# 添加数值标签
for i, v in enumerate(formatted_df['实际值']):
    plt.text(i - width/2, v + 0.3, f"{v:.1f}", ha='center', fontsize=9)
    
for i, v in enumerate(formatted_df['预测值']):
    plt.text(i + width/2, v + 0.3, f"{v:.1f}", ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('eight_samples_comparison.png', dpi=300, bbox_inches='tight')
print("8组样本对比图已保存到 eight_samples_comparison.png") 