#!/usr/bin/env python3
import unittest
import requests
import json
import time
import os

# 等待后端服务器启动
time.sleep(2)

API_URL = "http://localhost:9001"

class ApiTestExtended(unittest.TestCase):
    """扩展API测试，包含认证和数据管理接口测试"""
    
    def setUp(self):
        """每个测试前的设置"""
        # 存储登录后的会话和令牌
        self.session = requests.Session()
        self.auth_token = None
        
        # 测试数据文件路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.test_data_file = os.path.join(current_dir, "test_data", "sample_data.csv")
    
    def test_1_api_status(self):
        """测试API状态端点"""
        response = self.session.get(f"{API_URL}/api/status")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertTrue(data["database_connected"])
        self.assertTrue(all(data["models_loaded"].values()))

    def test_2_login(self):
        """测试登录功能"""
        login_data = {
            "username": "admin",  # 测试用户名
            "password": "admin"   # 测试密码
        }
        
        response = self.session.post(f"{API_URL}/api/login", json=login_data)
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertTrue(data["success"])
        self.assertIsNotNone(data["token"])
        
        # 保存认证令牌以用于后续测试
        self.auth_token = data["token"]
        self.session.headers.update({"Authorization": f"Bearer {self.auth_token}"})

    def test_3_invalid_login(self):
        """测试无效登录尝试"""
        login_data = {
            "username": "wrong_user",
            "password": "wrong_pass"
        }
        
        response = requests.post(f"{API_URL}/api/login", json=login_data)
        self.assertEqual(response.status_code, 401)
        
        data = response.json()
        self.assertFalse(data["success"])
        self.assertIn("error", data)

    def test_4_user_profile(self):
        """测试获取用户资料"""
        # 确保已经登录
        if not self.auth_token:
            self.test_2_login()
            
        response = self.session.get(f"{API_URL}/api/user/profile")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["profile"]["username"], "admin")

    def test_5_prediction(self):
        """测试预测功能（带认证）"""
        # 确保已经登录
        if not self.auth_token:
            self.test_2_login()
            
        test_data = {
            "model_type": "yield_strength",
            "features": {
                "c": 0.15,
                "Si": 0.25,
                "Mn": 1.4,
                "ActualExitTemperature": 1210,
                "ActualFinalRollingThickness": 12,
                "TemperatureAfterCooling": 550
            }
        }
        
        response = self.session.post(f"{API_URL}/api/predict", json=test_data)
        print(f"预测响应: {response.text}")
        
        # 检查请求是否被接受
        self.assertTrue(response.status_code in [200, 400])
        
        data = response.json()
        if data.get("success", False):
            if "prediction" in data:
                self.assertTrue(isinstance(data["prediction"], (int, float)))

    def test_6_analyze_models(self):
        """测试模型分析接口"""
        # 确保已经登录
        if not self.auth_token:
            self.test_2_login()
            
        response = self.session.get(f"{API_URL}/api/analyze-models")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertTrue(data.get("success", False))
        self.assertTrue("results" in data)
        
        # 检查返回的模型信息
        self.assertTrue("elongation" in data["results"])
        self.assertTrue("feature_names" in data["results"]["elongation"])

    def test_7_data_history(self):
        """测试获取预测历史数据"""
        # 确保已经登录
        if not self.auth_token:
            self.test_2_login()
            
        response = self.session.get(f"{API_URL}/api/data/history")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertTrue(data["success"])
        self.assertIsInstance(data["history"], list)

    def test_8_upload_data(self):
        """测试数据上传功能"""
        # 确保已经登录
        if not self.auth_token:
            self.test_2_login()
        
        # 如果测试数据文件存在则测试上传
        if os.path.exists(self.test_data_file):
            with open(self.test_data_file, 'rb') as file:
                files = {'file': file}
                response = self.session.post(f"{API_URL}/api/data/upload", files=files)
                
                self.assertEqual(response.status_code, 200)
                data = response.json()
                self.assertTrue(data["success"])
                self.assertIn("records", data)
        else:
            print(f"警告：测试数据文件 {self.test_data_file} 不存在，跳过上传测试")

    def test_9_logout(self):
        """测试登出功能"""
        # 确保已经登录
        if not self.auth_token:
            self.test_2_login()
            
        response = self.session.post(f"{API_URL}/api/logout")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertTrue(data["success"])
        
        # 尝试访问需要认证的接口，应该失败
        profile_response = self.session.get(f"{API_URL}/api/user/profile")
        self.assertEqual(profile_response.status_code, 401)

if __name__ == "__main__":
    unittest.main() 