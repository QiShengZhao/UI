import pandas as pd
import numpy as np
import joblib
import sys

def load_model():
    """加载已保存的模型和缩放器"""
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
    
    return model, scaler, model_version

def get_feature_names():
    """获取特征名称"""
    # 读取样本数据以获取特征名称
    df = pd.read_csv('test1.csv')
    feature_columns = [col for col in df.columns if col not in ['实测1', '实测2', '实测3']]
    return feature_columns

def predict_from_input(model, scaler, feature_names):
    """根据用户输入进行预测"""
    print("\n请按提示输入特征值（输入 'q' 退出）：")
    
    # 创建一个空字典来存储输入值
    input_data = {}
    
    for feature in feature_names:
        while True:
            value = input(f"请输入 {feature} 的值: ")
            if value.lower() == 'q':
                print("退出程序")
                sys.exit(0)
            
            try:
                input_data[feature] = float(value)
                break
            except ValueError:
                print("请输入有效的数值")
    
    # 转换为DataFrame
    input_df = pd.DataFrame([input_data])
    
    # 缩放特征
    input_scaled = scaler.transform(input_df)
    
    # 进行预测
    prediction = model.predict(input_scaled)[0]
    
    print(f"\n预测的韧性值: {prediction:.2f} J/cm²")
    return prediction

def predict_from_csv(model, scaler, csv_file):
    """从CSV文件预测韧性值"""
    try:
        # 读取CSV文件
        input_df = pd.read_csv(csv_file)
        print(f"已加载 {len(input_df)} 行数据")
        
        # 缩放特征
        input_scaled = scaler.transform(input_df)
        
        # 进行预测
        predictions = model.predict(input_scaled)
        
        # 添加预测结果列
        input_df['预测韧性值'] = predictions
        
        # 保存结果
        output_file = 'predictions_output.csv'
        input_df.to_csv(output_file, index=False)
        print(f"预测结果已保存到 {output_file}")
        
        # 显示前5个预测结果
        print("\n前5个预测结果:")
        for i in range(min(5, len(predictions))):
            print(f"样本 {i+1}: 预测韧性值 = {predictions[i]:.2f} J/cm²")
            
    except Exception as e:
        print(f"预测时出错: {str(e)}")

def main():
    # 加载模型
    model, scaler, model_version = load_model()
    print(f"已加载{model_version}随机森林模型")
    
    # 获取特征名称
    feature_names = get_feature_names()
    
    # 确定操作模式
    while True:
        print("\n请选择操作：")
        print("1. 输入单个样本进行预测")
        print("2. 从CSV文件批量预测")
        print("3. 退出")
        
        choice = input("请输入选项 (1-3): ")
        
        if choice == '1':
            predict_from_input(model, scaler, feature_names)
        elif choice == '2':
            csv_file = input("请输入CSV文件路径: ")
            predict_from_csv(model, scaler, csv_file)
        elif choice == '3':
            print("退出程序")
            break
        else:
            print("无效选项，请重新输入")

if __name__ == "__main__":
    main() 