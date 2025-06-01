from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os
import sys
import sqlite3
from pathlib import Path
import time
import json
from datetime import datetime

# 数据库配置
class db_connection:
    """数据库连接配置"""
    DATABASE = 'steel_system.db'

def get_db():
    """获取数据库连接"""
    conn = sqlite3.connect(db_connection.DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def initialize_db():
    """初始化数据库和表结构"""
    print(f"初始化数据库: {db_connection.DATABASE}")
    conn = get_db()
    cursor = conn.cursor()
    
    # 创建用户表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT DEFAULT 'user',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP
        )
    ''')
    
    # 创建预测历史表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            input_params TEXT NOT NULL,
            yield_strength REAL,
            tensile_strength REAL,
            elongation REAL,
            toughness REAL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    
    # 创建模型配置表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_config (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT NOT NULL,
            version TEXT NOT NULL,
            file_path TEXT NOT NULL,
            accuracy REAL,
            created_by INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_active BOOLEAN DEFAULT 1,
            FOREIGN KEY (created_by) REFERENCES users(id)
        )
    ''')
    
    # 创建默认管理员账户
    cursor.execute("SELECT * FROM users WHERE username = ?", ('admin',))
    if not cursor.fetchone():
        cursor.execute(
            "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
            ('admin', 'admin123', 'admin')
        )
    
    conn.commit()
    print("数据库初始化完成")

app = Flask(__name__)
CORS(app)  # 启用跨域请求

# 全局变量，用于存储加载的模型
models = {
    'toughness': {'model': None, 'scaler': None},
    'yield_strength': {'model': None, 'scaler': None},
    'tensile_strength': {'model': None, 'scaler': None},
    'elongation': {'model': None, 'scaler': None}
}

# 特征名称映射
FEATURE_MAPPINGS = {
    'yield_strength': {
        'C': 'C', 
        'Si': 'Si', 
        'Mn': 'Mn',
        'P': 0.02,
        'S': 0.01,
        'Alt': 0.03,
        'Cr': 0.01,
        'Cu': 0.01,
        'V': 0.005,
        'Nb': 0.005,
        'Mo': 0.01,
        'Ti': 0.002,
        'ModelCalculatedValue': 450,
        'TheoreticalSlabThickness': 230,
        'TheoreticalSlabWidth': 1250,
        'TimeinFurnace(min)': 50,
        '出炉温度实际值': 'ActualExitTemperature',
        'ActualRollingStartTemperature': 1220,
        'ActualStartAfterTemperatureControl': 990,
        '终扎厚度实际值': 'ActualFinalRollingThickness',
        'ActualFinalRollingTemperature': 880,
        'ActualTemperatureBeforeCooling': 820,
        'Thickness': 70,  
        '水冷后温度': 'TemperatureAfterCooling'
    },
    'tensile_strength': {
        'C': 'C', 
        'Si': 'Si', 
        'Mn': 'Mn',
        'P': 0.02,
        'S': 0.01,
        'Alt': 0.03,
        'Cr': 0.01,
        'Cu': 0.01,
        'V': 0.005,
        'Nb': 0.005,
        'Mo': 0.01,
        'Ti': 0.002,
        'ModelCalculatedValue': 550,
        'TheoreticalSlabThickness': 230,
        'TheoreticalSlabWidth': 1250,
        'TimeinFurnace(min)': 50,
        '出炉温度实际值': 'ActualExitTemperature',
        'ActualRollingStartTemperature': 1220,
        'ActualStartAfterTemperatureControl': 990,
        '终扎厚度实际值': 'ActualFinalRollingThickness',
        'ActualFinalRollingTemperature': 880,
        'ActualTemperatureBeforeCooling': 820,
        'Thickness': 70,  
        '水冷后温度': 'TemperatureAfterCooling'
    },
    'toughness': {
        'C': 'C',
        'Si': 'Si',
        'Mn': 'Mn',
        'P': 0.02,
        'S': 0.01,
        'Alt': 0.03,
        'Cr': 0.01,
        'Cu': 0.01,
        'V': 0.005,
        'Nb': 0.005,
        'Mo': 0.01,
        'Ti': 0.002,
        '模型计算值': 180,
        '钢坯理论厚度': 230,
        '钢坯理论宽度': 1250,
        '在炉时间(分钟)': 50,
        '出炉温度实际值': '出炉温度实际值',
        '开扎温度实际值': 1220,
        '控温后开扎实际值': 990,
        '终扎厚度实际值': '终扎厚度实际值',
        '终扎温度实际测量值': 880,
        '水冷前温度实际值': 820,
        '厚度': 70,
        '水冷后温度': '水冷后温度',
        '屈服强度': 385,
        '抗拉强度': 540,
        '实际%': 30
    },
    'elongation': {
        'C': 'C',
        'Si': 'Si',
        'Mn': 'Mn',
        'P': 0.02,
        'S': 0.01,
        'Alt': 0.03,
        'Cr': 0.01,
        'Cu': 0.01,
        'V': 0.005,
        'Nb': 0.005,
        'Mo': 0.01,
        'Ti': 0.002,
        '模型计算值': 180,
        '钢坯理论厚度': 230,
        '钢坯理论宽度': 1250,
        '在炉时间(分钟)': 50,
        '出炉温度实际值': '出炉温度实际值',
        '开扎温度实际值': 1220,
        '控温后开扎实际值': 990,
        '终扎厚度实际值': '终扎厚度实际值',
        '终扎温度实际测量值': 880,
        '水冷前温度实际值': 820,
        '厚度': 70,
        '水冷后温度': '水冷后温度',
        '屈服强度': 385,
        '抗拉强度': 540
    }
}

def analyze_model_features(model_type):
    """分析模型的特征要求"""
    if models[model_type]['model'] is None or models[model_type]['scaler'] is None:
        print(f"{model_type} 模型或缩放器未加载，无法分析特征")
        return None

    feature_names = None
    expected_feature_count = 0
    
    # 尝试获取模型期望的特征
    try:
        if model_type == 'elongation':
            # 对于随机森林模型，可以从n_features_in_属性获取特征数
            if hasattr(models[model_type]['model'], 'n_features_in_'):
                expected_feature_count = int(models[model_type]['model'].n_features_in_)
                print(f"{model_type} 模型期望的特征数量: {expected_feature_count}")
            
            # 尝试获取特征名称
            if hasattr(models[model_type]['scaler'], 'feature_names_in_'):
                # 将NumPy数组转换为Python列表，以便JSON序列化
                feature_names = models[model_type]['scaler'].feature_names_in_.tolist()
                print(f"{model_type} 模型的特征名称: {feature_names}")
    except Exception as e:
        print(f"分析 {model_type} 模型特征时出错: {str(e)}")
    
    return {
        'expected_feature_count': expected_feature_count,
        'feature_names': feature_names
    }

def prepare_input_for_model(model_type, input_data):
    """根据模型类型准备输入特征"""
    mapping = FEATURE_MAPPINGS[model_type].copy()
    prepared_data = {}
    
    # 处理动态输入
    for src_col, dst_col in mapping.items():
        if isinstance(dst_col, (int, float)):  # 如果是默认值
            prepared_data[src_col] = dst_col
        elif isinstance(dst_col, str):
            if dst_col in input_data:  # 如果是映射到输入数据中的某一列
                prepared_data[src_col] = input_data[dst_col]
            elif src_col in input_data:  # 如果源列名直接在输入数据中
                prepared_data[src_col] = input_data[src_col]
            else:
                # 处理中文字段到英文字段的映射
                # 对于复杂映射，直接检查特定的字段
                if dst_col == '出炉温度实际值' and '出炉温度实际值' in input_data:
                    prepared_data[src_col] = input_data['出炉温度实际值']
                elif dst_col == '终扎厚度实际值' and '终扎厚度实际值' in input_data:
                    prepared_data[src_col] = input_data['终扎厚度实际值']
                elif dst_col == '水冷后温度' and '水冷后温度' in input_data:
                    prepared_data[src_col] = input_data['水冷后温度']
                else:
                    prepared_data[src_col] = dst_col  # 保留映射中的值
        else:
            prepared_data[src_col] = dst_col
    
    # 特殊处理elongation模型 - 它需要特定的特征
    if model_type == 'elongation':
        prepared_data_df = pd.DataFrame([prepared_data])
        current_features = list(prepared_data.keys())
        
        # 添加特殊处理，确保伸长率模型有屈服强度和抗拉强度数据
        if '屈服强度' not in prepared_data and 'yield_strength' in input_data:
            prepared_data['屈服强度'] = float(input_data['yield_strength'])
            print(f"添加屈服强度: {input_data['yield_strength']}")
        elif '屈服强度' not in prepared_data:
            prepared_data['屈服强度'] = 385.0  # 默认值
            print("使用默认屈服强度: 385.0")
            
        if '抗拉强度' not in prepared_data and 'tensile_strength' in input_data:
            prepared_data['抗拉强度'] = float(input_data['tensile_strength'])
            print(f"添加抗拉强度: {input_data['tensile_strength']}")
        elif '抗拉强度' not in prepared_data:
            prepared_data['抗拉强度'] = 540.0  # 默认值
            print("使用默认抗拉强度: 540.0")
        
        # 获取elongation模型所需的特征列表
        expected_features = None
        if hasattr(models[model_type]['scaler'], 'feature_names_in_'):
            expected_features = models[model_type]['scaler'].feature_names_in_.tolist()
            print(f"伸长率模型期望的特征列表: {expected_features}")
        
        if expected_features:
            # 如果有我们缺少的特征，添加默认值
            for feature in expected_features:
                if feature not in prepared_data:
                    # 对于缺失特征，使用0作为默认值
                    prepared_data[feature] = 0
                    print(f"为伸长率模型添加缺失特征: {feature} = 0")
            
            # 删除模型不需要的特征
            extra_features = set(current_features) - set(expected_features)
            for feature in extra_features:
                if feature in prepared_data:
                    del prepared_data[feature]
                    print(f"删除伸长率模型不需要的特征: {feature}")
            
            # 确保特征顺序与模型期望的一致
            ordered_data = {}
            for feature in expected_features:
                if feature in prepared_data:
                    ordered_data[feature] = prepared_data[feature]
                else:
                    ordered_data[feature] = 0  # 添加默认值以确保所有特征都存在
                    print(f"在有序数据中添加缺失特征: {feature} = 0")
            prepared_data = ordered_data
    
    print(f"模型 {model_type} 最终准备的数据: {prepared_data}")
    print(f"特征数量: {len(prepared_data)}")
    
    # 将准备好的数据转换为DataFrame
    return pd.DataFrame([prepared_data])

def load_model(model_type, model_file, scaler_file):
    """加载指定的模型和缩放器"""
    try:
        models[model_type]['model'] = joblib.load(model_file)
        models[model_type]['scaler'] = joblib.load(scaler_file)
        print(f"成功加载 {model_type} 模型和缩放器")
        return True
    except Exception as e:
        print(f"加载 {model_type} 模型出错: {str(e)}")
        return False

# 路径映射
MODEL_PATHS = {
    'toughness': {
        'dir': '韧性',
        'model': 'best_rf_model_chinese.pkl',
        'scaler': 'rf_feature_scaler_chinese.pkl'
    },
    'yield_strength': {
        'dir': '屈服强度',
        'model': 'rf_model.pkl',
        'scaler': 'scaler.pkl'
    },
    'tensile_strength': {
        'dir': '抗拉强度',
        'model': 'xgb_model.pkl',
        'scaler': 'scaler.pkl'
    },
    'elongation': {
        'dir': '伸长率/模型',
        'model': 'rf_elongation_model.pkl',
        'scaler': 'rf_elongation_scaler.pkl'
    }
}

@app.route('/api/predict', methods=['POST'])
def predict():
    """接收预测请求并返回预测结果"""
    try:
        print(f"收到预测请求...")
        # 获取输入数据
        input_data = request.json
        print(f"输入数据: {input_data}")
        
        # 验证输入
        required_fields = ['c', 'si', 'mn', 'temp', 'thickness', 'coolTemp']
        for field in required_fields:
            if field not in input_data:
                error_msg = f'缺少必要的参数: {field}'
                print(f"错误: {error_msg}")
                return jsonify({
                    'success': False, 
                    'error': error_msg
                }), 400
        
        # 构建基础输入数据字典
        input_dict = {
            'C': float(input_data['c']),
            'Si': float(input_data['si']),
            'Mn': float(input_data['mn']),
            '出炉温度实际值': float(input_data['temp']),
            '终扎厚度实际值': float(input_data['thickness']),
            '水冷后温度': float(input_data['coolTemp']),
            # 添加英文键以支持不同模型
            'ActualExitTemperature': float(input_data['temp']),
            'ActualFinalRollingThickness': float(input_data['thickness']),
            'TemperatureAfterCooling': float(input_data['coolTemp'])
        }
        
        print(f"基础输入数据: {input_dict}")
        
        # 存储预测结果
        predictions = {}
        errors = {}
        
        # 首先预测屈服强度和抗拉强度，因为伸长率模型可能需要这些值
        prediction_order = ['yield_strength', 'tensile_strength', 'toughness', 'elongation']
        
        # 对每个模型类型按顺序进行预测
        for model_type in prediction_order:
            print(f"\n开始使用 {model_type} 模型进行预测...")
            if models[model_type]['model'] is not None and models[model_type]['scaler'] is not None:
                try:
                    # 更新输入数据，加入之前的预测结果
                    if model_type == 'elongation' and predictions:
                        for pred_type, pred_value in predictions.items():
                            input_dict[pred_type] = pred_value
                            # 对于伸长率模型，特别添加屈服强度和抗拉强度的中文字段
                            if pred_type == 'yield_strength':
                                input_dict['屈服强度'] = pred_value
                            elif pred_type == 'tensile_strength':
                                input_dict['抗拉强度'] = pred_value
                    
                    # 为此模型准备特定的输入数据
                    model_input = prepare_input_for_model(model_type, input_dict)
                    print(f"{model_type}模型输入数据:\n{model_input}")
                    
                    # 打印当前特征列
                    if hasattr(models[model_type]['scaler'], 'get_feature_names_out'):
                        feature_names = models[model_type]['scaler'].get_feature_names_out()
                        print(f"{model_type}模型的特征列: {feature_names}")
                    elif hasattr(models[model_type]['scaler'], 'feature_names_in_'):
                        feature_names = models[model_type]['scaler'].feature_names_in_
                        print(f"{model_type}模型的特征列: {feature_names}")
                    
                    # 打印输入数据列
                    print(f"输入数据列: {model_input.columns.tolist()}")
                    
                    # 检查特征列是否匹配
                    if hasattr(models[model_type]['scaler'], 'feature_names_in_'):
                        scaler_features = set(models[model_type]['scaler'].feature_names_in_)
                        input_features = set(model_input.columns)
                        
                        missing_features = scaler_features - input_features
                        extra_features = input_features - scaler_features
                        
                        if missing_features:
                            print(f"警告: 缺少特征: {missing_features}")
                            # 为缺少的特征添加默认值
                            for feature in missing_features:
                                model_input[feature] = 0
                        
                        if extra_features:
                            print(f"警告: 额外的特征: {extra_features}")
                            # 移除额外的特征
                            model_input = model_input.drop(columns=list(extra_features))
                        
                        # 确保列的顺序正确
                        model_input = model_input[models[model_type]['scaler'].feature_names_in_]
                    
                    try:
                        # 尝试缩放数据，捕获任何可能发生的错误
                        print(f"正在缩放 {model_type} 模型的输入数据...")
                        X_scaled = models[model_type]['scaler'].transform(model_input)
                        print(f"{model_type}缩放后数据形状: {X_scaled.shape}")
                    
                        # 尝试进行预测
                        print(f"正在预测 {model_type} ...")
                        prediction = models[model_type]['model'].predict(X_scaled)[0]
                        predictions[model_type] = float(prediction)
                        print(f"{model_type}预测结果: {prediction}")
                    except Exception as scale_err:
                        # 如果缩放或预测失败，记录详细错误
                        err_msg = f"{model_type}预测过程错误: {str(scale_err)}"
                        print(f"错误: {err_msg}")
                        errors[model_type] = err_msg
                        import traceback
                        traceback.print_exc()
                except Exception as e:
                    err_msg = f"{model_type}预测失败: {str(e)}"
                    print(f"错误: {err_msg}")
                    errors[model_type] = err_msg
                    import traceback
                    traceback.print_exc()
            else:
                err_msg = f"{model_type}模型或缩放器未加载"
                print(f"错误: {err_msg}")
                errors[model_type] = err_msg
        
        print(f"\n预测完成，结果: {predictions}")
        print(f"错误: {errors}")
        
        if not predictions:
            error_msg = "所有模型预测失败"
            print(f"错误: {error_msg}")
            print(f"详细错误: {errors}")
            return jsonify({
                'success': False,
                'error': error_msg,
                'details': errors
            }), 500
        
        print(f"所有预测完成: {predictions}")
        return jsonify({
            'success': True,
            'predictions': predictions
        })
        
    except Exception as e:
        error_msg = f"预测过程中出错: {str(e)}"
        print(f"错误: {error_msg}")
        import traceback
        traceback.print_exc()  # 打印完整堆栈跟踪
        return jsonify({
            'success': False,
            'error': str(e),
            'trace': traceback.format_exc()
        }), 500

@app.route('/api/status', methods=['GET'])
def status():
    """返回服务器状态和已加载的模型信息"""
    # 检查数据库连接
    db_status = False
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.fetchone()
        db_status = True
        conn.close()
    except Exception as e:
        print(f"数据库连接错误: {str(e)}")
    
    status_info = {
        'success': True,
        'status': 'API服务正常运行',
        'database_connected': db_status,
        'models_loaded': {
            model_type: models[model_type]['model'] is not None 
            for model_type in models
        }
    }
    return jsonify(status_info)

@app.route('/api/reload-models', methods=['POST'])
def reload_models():
    """重新加载所有模型"""
    results = {}
    
    for model_type, paths in MODEL_PATHS.items():
        model_path = os.path.join(paths['dir'], paths['model'])
        scaler_path = os.path.join(paths['dir'], paths['scaler'])
        
        results[model_type] = load_model(model_type, model_path, scaler_path)
    
    return jsonify({
        'success': True,
        'results': results
    })

def initialize_models():
    """初始化并加载所有模型"""
    print("开始初始化模型...")
    success_count = 0
    
    for model_type, paths in MODEL_PATHS.items():
        print(f"正在加载 {model_type} 模型...")
        try:
            model_path = os.path.join(paths['dir'], paths['model'])
            scaler_path = os.path.join(paths['dir'], paths['scaler'])
            
            print(f"模型路径: {model_path}")
            print(f"缩放器路径: {scaler_path}")
            
            if not os.path.exists(model_path):
                print(f"错误: 模型文件不存在: {model_path}")
                continue
                
            if not os.path.exists(scaler_path):
                print(f"错误: 缩放器文件不存在: {scaler_path}")
                continue
            
            success = load_model(model_type, model_path, scaler_path)
            if success:
                success_count += 1
                print(f"成功加载 {model_type} 模型")
            else:
                print(f"加载 {model_type} 模型失败")
        except Exception as e:
            print(f"加载 {model_type} 模型时发生错误: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print(f"模型初始化完成，成功加载 {success_count}/{len(MODEL_PATHS)} 个模型")

@app.route('/api/analyze-models', methods=['GET'])
def analyze_models():
    """分析所有已加载模型的特征要求"""
    results = {}
    
    for model_type in models:
        if models[model_type]['model'] is not None:
            results[model_type] = analyze_model_features(model_type)
        else:
            results[model_type] = {'loaded': False}
    
    return jsonify({
        'success': True,
        'results': results
    })

@app.route('/api/train-elongation-model', methods=['POST'])
def train_elongation_model():
    """训练新的伸长率模型"""
    try:
        print("开始训练新的伸长率模型...")
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import mean_squared_error, r2_score
        import time
        
        # 检查数据文件是否存在
        data_file = '伸长率/combined_data.csv'
        if not os.path.exists(data_file):
            return jsonify({
                'success': False,
                'error': f'数据文件不存在: {data_file}'
            }), 404
        
        # 读取数据
        print(f"正在读取数据: {data_file}")
        df = pd.read_csv(data_file)
        print(f"数据形状: {df.shape}")
        
        # 数据预处理
        X = df.drop('实际%', axis=1)  # 伸长率是目标变量
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
        start_time = time.time()
        rf_model.fit(X_train_scaled, y_train)
        training_time = time.time() - start_time
        print(f"模型训练完成，耗时 {training_time:.2f} 秒")
        
        # 预测测试集并评估模型
        y_pred = rf_model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mse)
        print(f"MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
        
        # 保存模型和缩放器
        model_dir = '伸长率/模型'
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, 'rf_elongation_model.pkl')
        scaler_path = os.path.join(model_dir, 'rf_elongation_scaler.pkl')
        
        print(f"保存模型到: {model_path}")
        joblib.dump(rf_model, model_path)
        print(f"保存缩放器到: {scaler_path}")
        joblib.dump(scaler, scaler_path)
        
        # 在内存中加载模型和缩放器
        models['elongation']['model'] = rf_model
        models['elongation']['scaler'] = scaler
        
        # 获取特征名称
        if hasattr(models['elongation']['scaler'], 'feature_names_in_'):
            feature_names = models['elongation']['scaler'].feature_names_in_.tolist()
        else:
            feature_names = X.columns.tolist()
        
        return jsonify({
            'success': True,
            'metrics': {
                'mse': float(mse),
                'rmse': float(rmse),
                'r2': float(r2),
                'training_time': float(training_time)
            },
            'feature_count': len(feature_names),
            'features': feature_names
        })
        
    except Exception as e:
        error_msg = f"训练伸长率模型时出错: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': error_msg
        }), 500

if __name__ == '__main__':
    print("初始化模型...")
    initialize_models()
    
    # 确定运行端口
    port = int(os.environ.get('PORT', 9001))
    
    print(f"启动服务器在端口 {port}...")
    app.run(host='0.0.0.0', port=port, debug=True) 