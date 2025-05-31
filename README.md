# 钢铁材料数据分析系统

本项目是一个基于 **HTML**、**JavaScript** 和 **Bootstrap** 的钢铁材料数据分析系统，支持数据上传、分析、可视化以及报告生成。

## 功能特性

- **数据上传**：支持上传 CSV 文件，自动解析并加载数据。
- **数据筛选**：根据元素含量、强度范围、厚度范围等条件筛选数据。
- **统计分析**：计算平均屈服强度、抗拉强度、延伸率和韧性值。
- **图表可视化**：
  - 元素含量与抗拉强度关系图
  - 工艺参数与强度关系图
  - 强度分布图
  - 厚度分布图
  - 热处理参数分布图
  - 相关性热图
  - 因素影响分析图
  - 韧性与成分关系图
- **AI 分析**：集成 DeepSeek API，提供智能分析与优化建议。
- **报告生成**：支持生成 PDF 或 HTML 格式的分析报告。

## 项目结构

```
.
├── index.html          # 主页面
├── main.js             # 核心逻辑与功能实现
├── test1.csv           # 示例数据文件
```

## 使用方法

### 1. 环境准备

确保您的浏览器支持以下库：
- [Bootstrap 5](https://getbootstrap.com/)
- [ECharts](https://echarts.apache.org/)
- [PapaParse](https://www.papaparse.com/)
- [html2pdf.js](https://github.com/eKoopmans/html2pdf)

### 2. 启动项目

1. 克隆项目到本地：
   ```bash
   git clone <仓库地址>
   cd <项目目录>
   ```

2. 打开 `index.html` 文件即可在浏览器中运行。

### 3. 功能操作

1. **上传数据**：
   - 点击导航栏中的“上传数据”按钮，选择 CSV 文件上传。
   - 示例数据文件为 `test1.csv`。

2. **数据筛选**：
   - 使用筛选器选择元素、强度范围或厚度范围，点击“应用筛选”。

3. **查看图表**：
   - 在“数据分析”页面查看生成的图表。

4. **生成报告**：
   - 点击“下载报告”按钮，选择报告内容和格式，生成并下载分析报告。

## 示例截图

### 数据分析页面
![数据分析页面](https://via.placeholder.com/800x400?text=数据分析页面)

### 报告生成页面
![报告生成页面](https://via.placeholder.com/800x400?text=报告生成页面)

## 技术栈

- **前端框架**：Bootstrap 5
- **数据解析**：PapaParse
- **图表库**：ECharts
- **文件导出**：html2pdf.js

## 贡献指南

欢迎对本项目提出建议或贡献代码：
1. Fork 本仓库。
2. 创建新分支：`git checkout -b feature/新功能描述`。
3. 提交更改：`git commit -m '添加新功能'`。
4. 推送分支：`git push origin feature/新功能描述`。
5. 提交 Pull Request。

## 开发者

- **飞飞** - 天津中德应用技术大学 - 机械工程学院

## 许可证

本项目采用 [MIT License](LICENSE) 许可证。