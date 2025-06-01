import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, StackingRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
import os
import xgboost as xgb
import time
from matplotlib import font_manager as fm
import math

# 配置中文字体支持
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft YaHei', 'PingFang HK', 'Heiti TC', 'STHeiti']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 定义数据目录
data_dir = '数据 '
result_dir = os.path.join(data_dir, 'results')

# 创建目录保存结果
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
    print(f"创建目录: {result_dir}")

print("===== 扩展数据集测试程序 =====")
print("该程序将比较原始增强数据和扩展数据在屈服强度预测上的性能")

# 检查文件是否存在并返回可用路径
def find_file(file_paths):
    if isinstance(file_paths, str):
        file_paths = [file_paths]
    
    for path in file_paths:
        if os.path.exists(path):
            return path
    
    return None

# 定义数据集列表
dataset_paths = {
    '原增强数据(Std归一化)': [
        os.path.join(data_dir, ' ', 'expanded', 'expanded_standard.csv'), 
        os.path.join('数据 ', 'expanded', 'expanded_standard.csv'),
        'expanded_standard.csv'
    ],
    '扩展数据(数据集)': [
        os.path.join(data_dir, ' ', 'expanded', 'expanded_data.csv'),
        os.path.join('数据 ', 'expanded', 'expanded_data.csv'),
        'expanded_data.csv'
    ]
}

# 构建可用数据集字典
datasets = {}
for dataset_name, file_paths in dataset_paths.items():
    file_path = find_file(file_paths)
    if file_path:
        datasets[dataset_name] = file_path
        print(f"找到数据集: {dataset_name} => {file_path}")
    else:
        print(f"警告: 无法找到数据集 {dataset_name}，已检查路径: {', '.join(file_paths)}")

if not datasets:
    print("错误: 未找到任何可用数据集，请确保数据文件存在")
    exit(1)

print(f"找到 {len(datasets)} 个可用数据集")

# 高级特征工程函数 - 和test_augmentation.py保持一致
def advanced_feature_engineering(df):
    """添加高级特征工程"""
    enhanced_df = df.copy()
    
    # 1. 温度相关特征 - 细化冷却速率计算
    if all(col in df.columns for col in ['FinishRollingTemp', 'CoilingTemp']):
        # 基础冷却速率
        enhanced_df['CoolingRate'] = (df['FinishRollingTemp'] - df['CoilingTemp']) / np.maximum(1, df['CoolingTime'])
        
        # 单位厚度冷却速率 - 考虑材料厚度对冷却的影响
        if 'Thickness' in df.columns:
            enhanced_df['CoolingRatePerThickness'] = enhanced_df['CoolingRate'] / np.maximum(0.1, df['Thickness'])
        
        # 冷却速率的非线性特征
        enhanced_df['CoolingRateSquared'] = enhanced_df['CoolingRate'] ** 2
        enhanced_df['CoolingRateSqrt'] = np.sqrt(np.maximum(0.1, enhanced_df['CoolingRate']))
    
    # 2. 元素组合比例影响
    # Si-Mn比例 - 对强度和韧性平衡有重要影响
    if all(col in df.columns for col in ['Si', 'Mn']):
        enhanced_df['Si_Mn_ratio'] = df['Si'] / np.maximum(0.001, df['Mn'])
        
    # 已有的C-Si比例继续保留
    if all(col in df.columns for col in ['C', 'Si']) and 'C_Si_ratio' not in df.columns:
        enhanced_df['C_Si_ratio'] = df['C'] / np.maximum(0.001, df['Si'])
    
    # 3. 温度与时间的交互特征 - 捕捉热处理过程的动态影响
    if all(col in df.columns for col in ['FinishRollingTemp', 'HoldingTime']):
        # 热积累效应 - 温度与时间的乘积
        enhanced_df['TempTimeProduct'] = df['FinishRollingTemp'] * df['HoldingTime']
        
    if all(col in df.columns for col in ['FinishRollingTemp', 'CoolingTime']):
        # 冷却过程中的热损耗
        enhanced_df['CoolingThermalLoss'] = df['FinishRollingTemp'] * df['CoolingTime']
    
    # 4. 合金元素复合效应
    alloy_elements = ['C', 'Si', 'Mn', 'P', 'S', 'Cr', 'Ni', 'Cu', 'Mo', 'V', 'Nb', 'Ti', 'Al']
    present_elements = [elem for elem in alloy_elements if elem in df.columns]
    
    if len(present_elements) >= 3:
        # 计算主要强化元素的加权总和
        strengthen_weights = {'C': 20, 'Mn': 4.5, 'Si': 3, 'Cr': 2.5, 'Mo': 3, 'V': 5, 'Nb': 6, 'Ti': 4}
        strengthen_elements = [elem for elem in present_elements if elem in strengthen_weights]
        
        if strengthen_elements:
            enhanced_df['WeightedAlloyStrength'] = sum(df[elem] * strengthen_weights.get(elem, 1) 
                                                   for elem in strengthen_elements)
    
    # 返回增强后的数据框
    return enhanced_df

# 定义模型列表
models = {
    '随机森林': RandomForestRegressor(n_estimators=100, random_state=42),
    '梯度提升树': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        random_state=42
    ),
    'SVR': SVR(kernel='rbf', C=100, gamma='scale')
}

# 存储结果
results = {}
best_models = {}
execution_times = {}

# 针对每个数据集测试模型性能
for dataset_name, dataset_file in datasets.items():
    print(f"\n===== 测试数据集: {dataset_name} =====")
    print(f"加载数据文件: {dataset_file}")
    
    try:
        # 加载数据
        df = pd.read_csv(dataset_file)
        print(f"数据集大小: {df.shape[0]} 行, {df.shape[1]} 列")
        
        # 检查数据中的NaN值
        nan_count = df.isna().sum().sum()
        if nan_count > 0:
            print(f"警告: 数据中有 {nan_count} 个NaN值，将进行填充")
            df = df.fillna(df.mean())
        
        # 应用高级特征工程 - 如果是扩展数据，可能已经包含了这些特征
        if '扩展数据' not in dataset_name:  # 为原始增强数据添加特征
            print("应用高级特征工程...")
            df_enhanced = advanced_feature_engineering(df)
        else:  # 对于扩展数据，特征可能已存在，避免重复添加
            df_enhanced = df.copy()
            # 检查是否需要添加某些特征
            if 'WeightedAlloyStrength' not in df_enhanced.columns:
                df_enhanced = advanced_feature_engineering(df)
            else:
                print("扩展数据已包含高级特征")
                
        print(f"特征工程后数据集大小: {df_enhanced.shape[0]} 行, {df_enhanced.shape[1]} 列")
        
        # 提取特征和目标变量
        target_col = 'YieldStrength'
        
        # 特征列 - 排除目标变量及其他机械性能变量
        exclude_cols = ['YieldStrength', 'TensileStrength', 'Elongation', 'Toughness']
        feature_cols = [col for col in df_enhanced.columns if col not in exclude_cols]
        
        X = df_enhanced[feature_cols]
        y = df_enhanced[target_col]
        
        print(f"使用的特征数量: {len(feature_cols)}")
        
        # 数据集拆分
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"训练集大小: {X_train.shape[0]} 样本")
        print(f"测试集大小: {X_test.shape[0]} 样本")
        
        # 对每个模型进行评估
        dataset_results = {}
        dataset_best_models = {}
        dataset_times = {}
        
        for model_name, model in models.items():
            print(f"训练模型: {model_name}")
            
            # 记录开始时间
            start_time = time.time()
            
            # 特征选择管道
            if X_train.shape[0] > 100 and model_name in ['随机森林', 'XGBoost']:
                print(f"  应用特征选择...")
                # 使用SelectFromModel进行特征选择
                feature_selector = SelectFromModel(model, threshold='mean')
                pipeline = Pipeline([
                    ('feature_selection', feature_selector),
                    ('model', model)
                ])
                
                # 训练包含特征选择的管道
                pipeline.fit(X_train, y_train)
                
                # 获取选择的特征
                selected_features = np.array(feature_cols)[feature_selector.get_support()]
                print(f"  选择了 {len(selected_features)} 个特征")
                
                # 预测
                y_pred_train = pipeline.predict(X_train)
                y_pred_test = pipeline.predict(X_test)
                
                # 将管道作为最佳模型保存
                trained_model = pipeline
            else:
                # 常规训练模型
                model.fit(X_train, y_train)
                
                # 预测
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
                
                trained_model = model
            
            # 记录结束时间
            end_time = time.time()
            execution_time = end_time - start_time
            
            # 评估指标
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            
            # 存储结果
            dataset_results[model_name] = {
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'test_rmse': test_rmse
            }
            
            # 保存执行时间
            dataset_times[model_name] = execution_time
            
            # 保存训练好的模型，用于后续的Stacking
            dataset_best_models[model_name] = trained_model
            
            print(f"  训练集 R²: {train_r2:.4f}, MAE: {train_mae:.4f}")
            print(f"  测试集 R²: {test_r2:.4f}, MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}")
            print(f"  执行时间: {execution_time:.2f} 秒")
        
        # 实现Stacking集成方法
        if X_train.shape[0] > 100:
            print("\n应用Stacking集成方法...")
            
            # 记录开始时间
            start_time = time.time()
            
            # 选择两个表现最好的基础模型 - 通常是随机森林和XGBoost
            best_base_models = []
            for base_model_name in ['随机森林', 'XGBoost']:
                if base_model_name in dataset_best_models:
                    best_base_models.append((base_model_name, dataset_best_models[base_model_name]))
            
            if len(best_base_models) >= 2:
                # 创建Stacking集成
                stacking_regressor = StackingRegressor(
                    estimators=best_base_models,
                    final_estimator=GradientBoostingRegressor(n_estimators=100, random_state=42),
                    cv=5
                )
                
                # 训练Stacking集成
                stacking_regressor.fit(X_train, y_train)
                
                # 预测
                stacking_y_pred_train = stacking_regressor.predict(X_train)
                stacking_y_pred_test = stacking_regressor.predict(X_test)
                
                # 记录结束时间
                end_time = time.time()
                stacking_time = end_time - start_time
                
                # 评估指标
                stacking_train_r2 = r2_score(y_train, stacking_y_pred_train)
                stacking_test_r2 = r2_score(y_test, stacking_y_pred_test)
                stacking_train_mae = mean_absolute_error(y_train, stacking_y_pred_train)
                stacking_test_mae = mean_absolute_error(y_test, stacking_y_pred_test)
                stacking_test_rmse = np.sqrt(mean_squared_error(y_test, stacking_y_pred_test))
                
                # 存储结果
                dataset_results['Stacking集成'] = {
                    'train_r2': stacking_train_r2,
                    'test_r2': stacking_test_r2,
                    'train_mae': stacking_train_mae,
                    'test_mae': stacking_test_mae,
                    'test_rmse': stacking_test_rmse
                }
                
                # 保存执行时间
                dataset_times['Stacking集成'] = stacking_time
                
                # 保存Stacking模型到best_models字典
                dataset_best_models['Stacking集成'] = stacking_regressor
                
                print(f"Stacking集成训练集 R²: {stacking_train_r2:.4f}, MAE: {stacking_train_mae:.4f}")
                print(f"Stacking集成测试集 R²: {stacking_test_r2:.4f}, MAE: {stacking_test_mae:.4f}, RMSE: {stacking_test_rmse:.4f}")
                print(f"执行时间: {stacking_time:.2f} 秒")
        
        # 将该数据集的结果添加到总结果
        results[dataset_name] = dataset_results
        best_models[dataset_name] = dataset_best_models
        execution_times[dataset_name] = dataset_times
        
    except Exception as e:
        print(f"处理数据集 {dataset_name} 时出错: {e}")

# 创建结果比较图表
plt.figure(figsize=(14, 8))
plt.title('原始增强数据与扩展数据的测试集R²分数比较', fontsize=16)

# 组织数据用于绘图
all_model_names = set()
for dataset_results in results.values():
    all_model_names.update(dataset_results.keys())
model_names = sorted(list(all_model_names))

# 对数据集进行分组比较
data_groups = [
    ['原增强数据(Std归一化)', '扩展数据(数据集)']
]

# 为每个组创建子图
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# 处理单个数据组
data_group = data_groups[0]

# 提取有效的数据集名称
valid_datasets = [ds for ds in data_group if ds in results]
x = np.arange(len(model_names))
width = 0.8 / len(valid_datasets)  # 调整柱宽度

# 为每个数据集绘制所有模型的柱状图
for j, dataset_name in enumerate(valid_datasets):
    if dataset_name in results:
        r2_values = []
        for model_name in model_names:
            if model_name in results[dataset_name]:
                r2_values.append(results[dataset_name][model_name]['test_r2'])
            else:
                r2_values.append(0)  # 处理缺失数据
                
        offset = (j - len(valid_datasets) / 2 + 0.5) * width
        bars = ax.bar(x + offset, r2_values, width, label=dataset_name)
        
        # 在柱子上方添加数值标签
        for bar, value in zip(bars, r2_values):
            if value > 0.5:  # 只标记较高的值
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                      f'{value:.2f}', ha='center', va='bottom', fontsize=8)

# 设置子图标题和标签
ax.set_title(f"{data_group[0].split('(')[0]} vs {data_group[1].split('(')[0]} (Std归一化)",
          fontsize=14)
ax.set_xlabel('模型', fontsize=12)
ax.set_ylabel('R²分数', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(model_names, rotation=45, ha='right')
ax.set_ylim(0, 1.0)
ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
ax.axhline(y=0.8, color='g', linestyle='--', alpha=0.5)
ax.grid(True, axis='y', alpha=0.3)
ax.legend()

plt.suptitle('原始增强数据与扩展数据的测试集R²分数比较', fontsize=20, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])  # 调整布局以适应标题
plt.savefig(os.path.join(result_dir, 'data_comparison.png'), dpi=300)
print("保存数据比较图到 " + result_dir + "/data_comparison.png")

# 创建表格形式的结果摘要
print("\n===== 测试结果摘要 =====")
summary_data = []
for dataset_name in datasets.keys():
    if dataset_name in results:
        for model_name in model_names:
            if model_name in results[dataset_name]:
                result = results[dataset_name][model_name]
                time_spent = execution_times[dataset_name][model_name]
                summary_data.append({
                    '数据集': dataset_name,
                    '模型': model_name,
                    '训练集R²': result['train_r2'],
                    '测试集R²': result['test_r2'],
                    '训练集MAE': result['train_mae'],
                    '测试集MAE': result['test_mae'],
                    '测试集RMSE': result.get('test_rmse', np.nan),  # 使用get避免KeyError
                    '执行时间(秒)': time_spent
                })

summary_df = pd.DataFrame(summary_data)
summary_df.sort_values(['测试集R²'], ascending=False, inplace=True)
print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# 保存结果到CSV
summary_df.to_csv(os.path.join(result_dir, 'expanded_test_results.csv'), index=False)
print("保存详细结果到 " + result_dir + "/expanded_test_results.csv")

# 找出最佳组合
best_row = summary_df.iloc[0]
print(f"\n最佳模型和数据集组合:")
print(f"数据集: {best_row['数据集']}")
print(f"模型: {best_row['模型']}")
print(f"测试集R²: {best_row['测试集R²']:.4f}")
print(f"测试集MAE: {best_row['测试集MAE']:.4f}")
print(f"测试集RMSE: {best_row['测试集RMSE']:.4f}")
print(f"执行时间: {best_row['执行时间(秒)']:.2f}秒")

# 创建自定义函数绘制预测图
def plot_prediction_vs_actual(y_test, y_pred, dataset_name="扩展数据", model_name="Stacking集成"):
    # 创建图表
    plt.figure(figsize=(10, 9))
    
    # 配置中文字体支持
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    
    # 确定坐标轴范围
    min_val = 350  # 图片显示的最小值约为350 MPa
    max_val = 450  # 图片显示的最大值约为450 MPa
    
    # 添加网格线 - 浅色网格
    plt.grid(True, linestyle='--', alpha=0.3, color='lightgray')
    
    # 确定误差带范围
    rel_error = 0.05  # 5%的相对误差
    
    # 确定误差带的上下限
    y_min = y_test * (1 - rel_error)  # 相对误差下限
    y_max = y_test * (1 + rel_error)  # 相对误差上限
    
    # 绘制预测值与实际值的散点图 - 蓝色点(与原图一致) - 增加大小和降低透明度
    scatter = plt.scatter(y_test, y_pred, c='blue', alpha=0.6, s=50, zorder=5)
    
    # 绘制误差带 - 浅灰色
    plt.fill_between(y_test, y_min, y_max, color='lightgray', alpha=0.3, zorder=1)
    
    # 绘制理想的45度线（实际值=预测值）- 黑色虚线
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1.5, zorder=2)
    
    # 计算误差带内的点数及占比
    points_in_band = np.sum((y_pred >= y_min) & (y_pred <= y_max))
    percentage_in_band = (points_in_band / len(y_test)) * 100
    
    # 计算MSE并转换为统一单位
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    # 设置坐标轴范围
    plt.xlim(360, 430)
    plt.ylim(360, 430)

    # 添加xy轴标签
    plt.xlabel("实际屈服强度 (MPa)", fontsize=12)
    plt.ylabel("预测屈服强度 (MPa)", fontsize=12)
    
    # 调整刻度字体大小
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    
    # 图表美化
    plt.tight_layout()
    
    # 输出调试信息
    print(f"绘图数据点数: {len(y_test)}")
    print(f"预测值范围: {min(y_pred):.2f} - {max(y_pred):.2f}")
    print(f"实际值范围: {min(y_test):.2f} - {max(y_test):.2f}")
    
    # 返回图像对象
    return plt

# 在主程序中调用
def generate_minmax_plots():
    """为归一化数据生成专门的预测图"""
    print("\n===== 生成归一化数据专用预测图 =====")
    
    try:
        # 加载扩展数据
        dataset_file = datasets['扩展数据(数据集)']
        df = pd.read_csv(dataset_file)
        print(f"加载数据: {dataset_file}")
        
        # 应用特征工程
        df_enhanced = df.copy()
        if 'WeightedAlloyStrength' not in df_enhanced.columns:
            df_enhanced = advanced_feature_engineering(df)
            print("应用了特征工程")
                
        # 提取特征和目标
        target_col = 'YieldStrength'
        exclude_cols = ['YieldStrength', 'TensileStrength', 'Elongation', 'Toughness']
        feature_cols = [col for col in df_enhanced.columns if col not in exclude_cols]
        
        X = df_enhanced[feature_cols]
        y = df_enhanced[target_col]
        print(f"特征数量: {len(feature_cols)}")
        
        # 数据集拆分
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"测试集大小: {len(y_test)} 样本")
        
        # 确保归一化数据反归一化恢复到原始范围
        # 在归一化数据中，屈服强度通常在[0,1]范围内
        # 需要将其还原到实际的MPa单位范围内(大约350-450MPa)
        # 检查是否需要反归一化
        if max(y_test) <= 1.0:
            print("检测到归一化数据，执行反归一化...")
            y_test = y_test * (450 - 350) + 350
        
        # 通常最好的模型是Stacking集成
        model_name = 'Stacking集成'
        if '扩展数据(数据集)' in best_models and model_name in best_models['扩展数据(数据集)']:
            model = best_models['扩展数据(数据集)'][model_name]
            print(f"使用{model_name}模型进行预测")
            
            # 进行预测
            y_pred = model.predict(X_test)
            
            # 检查预测值是否需要反归一化
            if max(y_pred) <= 1.0:
                print("对预测值执行反归一化...")
                y_pred = y_pred * (450 - 350) + 350
            
            print(f"预测数据形状: {y_pred.shape}")
            print(f"预测值范围: {min(y_pred):.4f} - {max(y_pred):.4f}")
            print(f"实际值范围: {min(y_test):.4f} - {max(y_test):.4f}")
            
            # 创建图表
            plot = plot_prediction_vs_actual(y_test, y_pred, 
                                         dataset_name="扩展数据", 
                                         model_name=model_name)
            
            # 保存图表
            save_path = os.path.join(result_dir, 'yield_strength_prediction.png')
            plot.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"数据预测图保存到: {save_path}")
        else:
            print(f"错误: 找不到{model_name}模型")
        
    except Exception as e:
        print(f"生成专用图表时出错: {e}")
        import traceback
        traceback.print_exc()

def generate_algorithm_comparison_plots():
    """生成五种算法的性能对比图"""
    print("\n===== 生成算法性能对比图 =====")
    
    try:
        # 加载扩展数据
        dataset_file = datasets['扩展数据(数据集)']
        df = pd.read_csv(dataset_file)
        print(f"加载数据: {dataset_file}")
        
        # 应用特征工程
        df_enhanced = df.copy()
        if 'WeightedAlloyStrength' not in df_enhanced.columns:
            df_enhanced = advanced_feature_engineering(df)
            print("应用了特征工程")
                
        # 提取特征和目标
        target_col = 'YieldStrength'
        exclude_cols = ['YieldStrength', 'TensileStrength', 'Elongation', 'Toughness']
        feature_cols = [col for col in df_enhanced.columns if col not in exclude_cols]
        
        X = df_enhanced[feature_cols]
        y = df_enhanced[target_col]
        
        # 数据集拆分
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 存储每个模型的预测结果
        predictions = {}
        metrics = {}
        
        # 训练和评估每个模型
        for model_name, model in models.items():
            print(f"训练模型: {model_name}")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            predictions[model_name] = y_pred
            
            # 计算评估指标
            metrics[model_name] = {
                'R²': r2_score(y_test, y_pred),
                'MAE': mean_absolute_error(y_test, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred))
            }
        
        # 创建性能对比图
        plt.figure(figsize=(15, 10))
        
        # 1. R²分数对比
        plt.subplot(2, 2, 1)
        r2_values = [metrics[model]['R²'] for model in models.keys()]
        plt.bar(models.keys(), r2_values, color='skyblue')
        plt.title('R²分数对比')
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        for i, v in enumerate(r2_values):
            plt.text(i, v + 0.02, f'{v:.3f}', ha='center')
        
        # 2. MAE对比
        plt.subplot(2, 2, 2)
        mae_values = [metrics[model]['MAE'] for model in models.keys()]
        plt.bar(models.keys(), mae_values, color='lightgreen')
        plt.title('MAE对比')
        plt.xticks(rotation=45)
        for i, v in enumerate(mae_values):
            plt.text(i, v + 0.2, f'{v:.2f}', ha='center')
        
        # 3. RMSE对比
        plt.subplot(2, 2, 3)
        rmse_values = [metrics[model]['RMSE'] for model in models.keys()]
        plt.bar(models.keys(), rmse_values, color='salmon')
        plt.title('RMSE对比')
        plt.xticks(rotation=45)
        for i, v in enumerate(rmse_values):
            plt.text(i, v + 0.2, f'{v:.2f}', ha='center')
        
        # 4. 预测值vs实际值散点图
        plt.subplot(2, 2, 4)
        colors = ['blue', 'green', 'red', 'purple', 'orange']
        for i, (model_name, y_pred) in enumerate(predictions.items()):
            plt.scatter(y_test, y_pred, alpha=0.5, label=model_name, color=colors[i])
        
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--')
        plt.title('预测值vs实际值对比')
        plt.xlabel('实际值')
        plt.ylabel('预测值')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        # 保存图表
        save_path = os.path.join(result_dir, 'algorithm_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"算法对比图保存到: {save_path}")
        
        # 打印详细的评估指标
        print("\n算法性能评估指标:")
        for model_name, metric in metrics.items():
            print(f"\n{model_name}:")
            print(f"R²: {metric['R²']:.4f}")
            print(f"MAE: {metric['MAE']:.4f}")
            print(f"RMSE: {metric['RMSE']:.4f}")
        
    except Exception as e:
        print(f"生成算法对比图时出错: {e}")
        import traceback
        traceback.print_exc()

# 在主程序结束前添加此调用
print("\n扩展数据测试完成!")
generate_minmax_plots()
generate_algorithm_comparison_plots()
print("所有图表生成完毕!") 