#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
伸长率预测模型 - 完整分析流程
运行此脚本将依次执行所有分析步骤，生成所有图表和报告
"""

import os
import subprocess
import time
import sys

def run_script(script_name, description):
    """运行指定的Python脚本并显示进度"""
    print(f"\n{'='*80}")
    print(f"执行: {description}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                               check=True, 
                               text=True, 
                               capture_output=True)
        print(result.stdout)
        if result.stderr:
            print(f"警告:\n{result.stderr}")
        
        print(f"\n完成: {description}")
        print(f"耗时: {time.time() - start_time:.2f} 秒")
        return True
    except subprocess.CalledProcessError as e:
        print(f"错误: 执行 {script_name} 失败")
        print(f"退出码: {e.returncode}")
        print(f"错误信息:\n{e.stderr}")
        return False

def main():
    """主函数，按顺序执行所有分析脚本"""
    print("\n伸长率预测模型 - 完整分析流程")
    print("="*50)
    
    # 检查所需文件是否存在
    required_files = ['combined_data.csv']
    for file in required_files:
        if not os.path.exists(file):
            print(f"错误: 找不到必需的文件 '{file}'")
            return False
    
    # 按顺序执行所有脚本
    scripts = [
        ('predict_elongation.py', '基本预测模型和可视化'),
        ('model_evaluation.py', '模型评估和高级分析'),
        ('show_top_predictions.py', '代表性样本分析'),
        ('generate_report.py', '生成综合报告')
    ]
    
    success = True
    for script, description in scripts:
        if not run_script(script, description):
            success = False
            print(f"\n警告: '{script}' 执行失败，但将继续执行后续脚本")
    
    if success:
        print("\n所有分析已成功完成！")
        
        # 列出生成的文件
        print("\n生成的文件:")
        output_files = [
            f for f in os.listdir('.') if f.endswith(('.png', '.pdf', '.html', '.csv'))
            and f != 'combined_data.csv' and os.path.getmtime(f) > time.time() - 3600
        ]
        
        for f in sorted(output_files):
            print(f"  - {f}")
    else:
        print("\n部分分析未能成功完成，请检查上述错误信息")
    
    return success

if __name__ == "__main__":
    main() 