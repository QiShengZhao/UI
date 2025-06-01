const axios = require('axios');
const MockAdapter = require('axios-mock-adapter');

// 首先，我们需要安装axios-mock-adapter作为开发依赖
// 这个测试会假设已经安装了axios-mock-adapter

describe('API Service', () => {
  let mock;
  
  beforeEach(() => {
    // 创建一个新的axios mock适配器
    mock = new MockAdapter(axios);
  });
  
  afterEach(() => {
    // 重置mock适配器
    mock.reset();
  });
  
  test('测试获取预测模型列表', async () => {
    // 模拟API响应
    const mockModels = [
      { id: 1, name: '抗拉强度预测模型', accuracy: 0.92 },
      { id: 2, name: '屈服强度预测模型', accuracy: 0.89 }
    ];
    
    // 设置mock响应
    mock.onGet('/api/models').reply(200, { models: mockModels });
    
    // 发起请求
    const response = await axios.get('/api/models');
    
    // 验证响应
    expect(response.status).toBe(200);
    expect(response.data.models).toEqual(mockModels);
  });
  
  test('测试提交预测数据', async () => {
    // 模拟请求数据
    const predictionData = {
      model_id: 1,
      features: {
        C: 0.35,
        Si: 0.28,
        Mn: 0.83,
        P: 0.015,
        S: 0.01,
        Cr: 0.9,
        Mo: 0.3
      }
    };
    
    // 模拟API响应
    const mockResult = {
      prediction: 780.5,
      confidence: 0.87
    };
    
    // 设置mock响应
    mock.onPost('/api/predict').reply(200, mockResult);
    
    // 发起请求
    const response = await axios.post('/api/predict', predictionData);
    
    // 验证响应
    expect(response.status).toBe(200);
    expect(response.data).toEqual(mockResult);
  });
  
  test('测试API错误处理', async () => {
    // 设置mock响应为错误
    mock.onGet('/api/models').reply(500, { error: '服务器错误' });
    
    // 使用try-catch捕获预期的错误
    try {
      await axios.get('/api/models');
      // 如果请求不抛出错误，则让测试失败
      expect(true).toBe(false);
    } catch (error) {
      // 验证错误
      expect(error.response.status).toBe(500);
      expect(error.response.data.error).toBe('服务器错误');
    }
  });
}); 