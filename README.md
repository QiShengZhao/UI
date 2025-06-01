# 钢铁工业分析系统

这是一个用于钢铁工业的综合分析和预测系统，基于机器学习模型预测钢材的机械性能，包括屈服强度、抗拉强度、伸长率和冲击韧性。

## 项目概述

本系统利用钢材的化学成分和工艺参数数据，通过机器学习模型进行分析和预测，帮助用户优化钢材生产过程，提高产品质量。系统支持单条数据预测和批量数据处理，并提供数据可视化分析功能。

### 主要功能

- **钢材性能预测**：基于化学成分和工艺参数预测钢材的机械性能
- **数据分析**：提供趋势分析、相关性分析、散点分析等多种数据可视化功能
- **批量处理**：支持CSV文件批量上传和预测
- **模型管理**：支持模型状态监控、特征分析和模型训练

### 技术栈

- **前端**：React, Ant Design, ECharts, Webpack
- **后端**：Flask, SQLite
- **机器学习**：随机森林、XGBoost等算法
- **数据处理**：Pandas, NumPy
- **测试**：Jest, React Testing Library, Cypress, Python单元测试

## 系统架构

```
├── 前端 (UI/)
│   ├── 页面组件 (src/pages/)
│   ├── 通用组件 (src/components/)
│   ├── 服务 (src/services/)
│   └── 工具 (src/utils/)
│
├── 后端 (server.py)
│   ├── API接口
│   ├── 数据库管理
│   └── 模型管理
│
└── 机器学习模型
    ├── 屈服强度模型
    ├── 抗拉强度模型
    ├── 伸长率模型
    └── 韧性模型
```

## 安装指南

### 前提条件

- Node.js 18.x 或更高版本
- Python 3.8 或更高版本
- pip 包管理器
- 操作系统：Windows, macOS 或 Linux

### 安装步骤

1. 克隆仓库
   ```bash
   git clone [仓库地址]
   cd 钢铁工业分析系统
   ```

2. 安装后端依赖
   ```bash
   pip install -r requirements.txt
   ```

3. 安装前端依赖
   ```bash
   cd UI
   npm install
   ```

## 使用指南

### 启动系统

1. 启动后端服务
   ```bash
   python server.py
   ```
   后端服务将在 http://localhost:9001 运行

2. 启动前端开发服务器
   ```bash
   cd UI
   npm start
   ```
   前端将在 http://localhost:9000 运行，并自动在浏览器中打开

### 登录系统

- 使用以下默认管理员账户登录：
  - 用户名：admin
  - 密码：admin123

### 使用功能

#### 钢材性能预测

1. **单条预测**：在"钢材性能预测"页面中，输入钢材的化学成分和工艺参数，点击"开始预测"按钮
2. **批量预测**：点击"批量预测"按钮，上传符合格式要求的CSV文件，点击"开始批量预测"按钮

#### 数据分析

1. 在"数据分析"页面中，可以上传自定义数据或使用系统默认数据
2. 选择要分析的性能指标（如屈服强度、抗拉强度等）和影响因子（如碳含量、硅含量等）
3. 通过趋势图、相关性热力图、散点图等方式查看数据分析结果

#### 模型管理（管理员功能）

1. 在"模型管理"页面中，可以查看各模型的状态和性能
2. 分析模型使用的特征和重要性
3. 训练和更新模型

### CSV文件格式要求

上传的CSV文件应包含以下字段：

- 化学成分数据（C, Si, Mn, P, S, Alt, Cr, Cu, V, Nb, Mo, Ti等）
- 工艺参数数据（出炉温度、钢板厚度、水冷温度等）

如果CSV文件解析出错，可使用提供的`csv_processor.py`工具修复：

```bash
python csv_processor.py
```

## 文件结构

```
├── UI/                       # 前端代码
│   ├── src/                  # 源代码
│   │   ├── components/       # 通用组件
│   │   ├── pages/            # 页面组件
│   │   ├── services/         # API服务
│   │   └── utils/            # 工具函数
│   ├── public/               # 静态资源
│   ├── test/                 # 测试文件
│   └── package.json          # 项目配置
│
├── 屈服强度/                  # 屈服强度模型
├── 抗拉强度/                  # 抗拉强度模型
├── 伸长率/                    # 伸长率模型
├── 韧性/                      # 韧性模型
│
├── server.py                 # Flask后端服务
├── csv_processor.py          # CSV处理工具
├── api_test.py               # API测试脚本
├── api_test_extended.py      # 扩展API测试
├── run_all_tests.py          # 测试运行脚本
├── requirements.txt          # Python依赖
└── steel_system.db           # SQLite数据库
```

## 开发指南

### 前端开发

1. 代码规范
   - 使用ES6+语法
   - 组件使用函数式组件和Hooks
   - 遵循React最佳实践

2. 添加新页面
   ```jsx
   // src/pages/NewPage.js
   import React from 'react';
   
   const NewPage = () => {
     return <div>新页面内容</div>;
   };
   
   export default NewPage;
   ```

3. 添加路由
   ```jsx
   // src/App.js
   import NewPage from './pages/NewPage';
   
   // 在路由配置中添加
   <Route path="/new-page" element={<NewPage />} />
   ```

### 后端开发

1. 添加新API端点
   ```python
   # server.py
   @app.route('/api/new-endpoint', methods=['GET', 'POST'])
   def new_endpoint():
       # 实现逻辑
       return jsonify({'status': 'success', 'data': result})
   ```

2. 数据库操作
   ```python
   # 示例：添加新记录
   def add_record():
       conn = get_db()
       cursor = conn.cursor()
       cursor.execute("INSERT INTO table_name (column1, column2) VALUES (?, ?)", 
                     (value1, value2))
       conn.commit()
   ```

## 测试框架

### 前端测试

1. 运行前端测试
   ```bash
   cd UI
   npm test
   ```

2. 运行端到端测试
   ```bash
   cd UI
   npm run cypress:open  # 打开Cypress测试运行器
   npm run cypress:run   # 在命令行中运行测试
   ```

### 后端测试

1. 运行API测试
   ```bash
   python api_test.py
   python api_test_extended.py
   ```

2. 运行所有测试
   ```bash
   python run_all_tests.py
   ```

## 常见问题解决

### CSV文件解析错误

问题：上传CSV文件时出现"Too few fields"错误

解决方案：
1. 确保CSV文件使用逗号作为分隔符
2. 检查文件编码是否为UTF-8
3. 使用提供的修复工具：`python csv_processor.py`

### 模型加载失败

问题：系统无法加载机器学习模型

解决方案：
1. 检查模型文件是否存在于正确的目录
2. 确保模型文件格式正确
3. 重启后端服务：`python server.py`

## 贡献指南

1. Fork 仓库
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建Pull Request

## 许可证

本项目采用 [LICENSE] 许可证 - 详情请参阅 LICENSE 文件。

## 联系方式

项目维护者 - [维护者邮箱]

项目链接: [项目仓库URL] 