#!/usr/bin/env python3
import os
import subprocess
import time
import unittest
from unittest import TestLoader, TextTestRunner
import sys

def run_backend_tests():
    """运行后端API测试"""
    print("\n=== 运行后端API测试 ===\n")
    
    # 导入API测试模块
    import api_test
    
    # 使用测试加载器运行测试
    test_loader = TestLoader()
    test_suite = test_loader.loadTestsFromModule(api_test)
    
    # 运行测试
    test_runner = TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    
    return result.wasSuccessful()

def run_extended_backend_tests():
    """运行扩展的后端API测试"""
    print("\n=== 运行扩展后端API测试 ===\n")
    
    try:
        # 导入扩展API测试模块
        import api_test_extended
        
        # 使用测试加载器运行测试
        test_loader = TestLoader()
        test_suite = test_loader.loadTestsFromModule(api_test_extended)
        
        # 运行测试
        test_runner = TextTestRunner(verbosity=2)
        result = test_runner.run(test_suite)
        
        return result.wasSuccessful()
    except ImportError:
        print("警告: api_test_extended.py 不存在，跳过扩展测试")
        return True

def run_frontend_tests():
    """运行前端测试"""
    print("\n=== 运行前端测试 ===\n")
    
    # 切换到UI目录
    os.chdir("UI")
    
    # 运行npm test命令
    process = subprocess.run(
        ["npm", "test"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # 打印输出
    print(process.stdout)
    if process.stderr:
        print(process.stderr)
    
    # 切回原目录
    os.chdir("..")
    
    return process.returncode == 0

def run_frontend_coverage():
    """运行前端测试覆盖率报告"""
    print("\n=== 运行前端测试覆盖率报告 ===\n")
    
    # 切换到UI目录
    os.chdir("UI")
    
    # 运行npm test命令带覆盖率
    process = subprocess.run(
        ["npm", "test", "--", "--coverage"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # 打印输出
    print(process.stdout)
    if process.stderr:
        print(process.stderr)
    
    # 切回原目录
    os.chdir("..")
    
    return process.returncode == 0

def run_e2e_tests():
    """运行端到端测试"""
    print("\n=== 运行端到端测试 ===\n")
    
    # 检查前端和后端服务是否已运行
    frontend_running = True  # 假设前端已在运行
    backend_running = True   # 假设后端已在运行
    
    # 切换到UI目录
    os.chdir("UI")
    
    # 运行E2E测试
    process = subprocess.run(
        ["npm", "run", "cypress:run"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # 打印输出
    print(process.stdout)
    if process.stderr:
        print(process.stderr)
    
    # 切回原目录
    os.chdir("..")
    
    return process.returncode == 0

def main():
    """主函数"""
    # 运行所有测试
    success_backend = run_backend_tests()
    success_extended_backend = run_extended_backend_tests()
    success_frontend = run_frontend_tests()
    
    # 如果基本测试都成功，则运行覆盖率和E2E测试
    if success_backend and success_frontend:
        success_coverage = run_frontend_coverage()
        if '--with-e2e' in sys.argv:
            success_e2e = run_e2e_tests()
        else:
            success_e2e = True
            print("\n跳过E2E测试。使用 '--with-e2e' 参数运行E2E测试")
    else:
        success_coverage = False
        success_e2e = True  # 不影响总体结果
    
    print("\n=== 测试结果汇总 ===")
    print(f"基本后端API测试: {'成功' if success_backend else '失败'}")
    print(f"扩展后端API测试: {'成功' if success_extended_backend else '失败'}")
    print(f"前端测试: {'成功' if success_frontend else '失败'}")
    print(f"前端覆盖率测试: {'成功' if success_coverage else '跳过或失败'}")
    if '--with-e2e' in sys.argv:
        print(f"端到端测试: {'成功' if success_e2e else '失败'}")
    
    # 设置退出码
    overall_success = success_backend and success_extended_backend and success_frontend
    sys.exit(0 if overall_success else 1)

if __name__ == "__main__":
    main() 