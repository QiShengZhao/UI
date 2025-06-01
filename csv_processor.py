#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import codecs
import os

def process_csv(input_file, output_file):
    """处理CSV文件，修复编码和分隔符问题"""
    print(f"处理文件: {input_file}")
    
    # 使用'utf-8'编码打开文件
    with codecs.open(input_file, 'r', encoding='utf-8') as infile:
        content = infile.read()
    
    # 确保使用正确的行分隔符
    if '\r\n' in content:
        print("检测到Windows风格的换行符，正在统一...")
        content = content.replace('\r\n', '\n')
    
    # 写入处理后的文件
    with codecs.open(output_file, 'w', encoding='utf-8') as outfile:
        outfile.write(content)
    
    print(f"文件处理完成，输出到: {output_file}")
    
    # 检验文件
    test_parse(output_file)
    
def test_parse(file_path):
    """测试解析CSV文件"""
    rows = []
    with codecs.open(file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            if i < 3:  # 只显示前3行
                print(f"第 {i+1} 行有 {len(row)} 列: {row[:5]}...")
            rows.append(row)
    
    print(f"成功读取 {len(rows)} 行数据")
    
if __name__ == "__main__":
    input_file = "test1.csv"
    output_file = "test1_fixed.csv"
    
    # 如果目标文件已存在，先删除
    if os.path.exists(output_file):
        os.remove(output_file)
    
    process_csv(input_file, output_file) 