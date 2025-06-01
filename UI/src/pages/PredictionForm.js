import React, { useState, useEffect } from 'react';
import { 
  Form, 
  Input, 
  Button, 
  Card, 
  Typography, 
  Row, 
  Col,
  Divider,
  Spin,
  message,
  Result,
  InputNumber,
  Tooltip,
  Tabs,
  Space,
  Table,
  Alert
} from 'antd';
import { 
  ExperimentOutlined, 
  LineChartOutlined, 
  HistoryOutlined,
  InfoCircleOutlined,
  FileOutlined,
  UserOutlined,
  DatabaseOutlined,
  CalculatorOutlined,
  CheckCircleOutlined
} from '@ant-design/icons';
import ReactECharts from 'echarts-for-react';
import ApiService from '../services/api';
import FileUploader from '../components/FileUploader';

const { Title, Text } = Typography;

const PredictionForm = () => {
  const [form] = Form.useForm();
  const [loading, setLoading] = useState(false);
  const [predictions, setPredictions] = useState(null);
  const [error, setError] = useState(null);
  const [predictionHistory, setPredictionHistory] = useState([]);
  const [activeTab, setActiveTab] = useState('input');
  const [batchMode, setBatchMode] = useState(false);
  const [batchData, setBatchData] = useState([]);
  const [batchResults, setBatchResults] = useState([]);
  const [batchPredicting, setBatchPredicting] = useState(false);

  const handleSubmit = async (values) => {
    setLoading(true);
    setError(null);
    setPredictions(null);
    
    try {
      // Transform the form values to match the API expectations
      const apiData = {
        c: values.carbon,
        si: values.silicon,
        mn: values.manganese,
        temp: values.exitTemperature,
        thickness: values.thickness,
        coolTemp: values.coolingTemperature
      };
      
      const response = await ApiService.makePrediction(apiData);
      
      if (response.success) {
        setPredictions(response.predictions);
        
        // Add to prediction history
        const historyItem = {
          id: Date.now(),
          date: new Date().toLocaleString(),
          ...apiData,
          ...response.predictions
        };
        
        setPredictionHistory(prev => [historyItem, ...prev.slice(0, 9)]);
        message.success('预测成功完成');
      } else {
        setError(response.error || '预测失败，请稍后再试');
        message.error('预测失败');
      }
    } catch (err) {
      setError(err.message || '服务器连接错误');
      message.error('服务器连接错误');
    } finally {
      setLoading(false);
    }
  };
  
  const resetForm = () => {
    form.resetFields();
    setPredictions(null);
    setError(null);
  };

  const getChartOption = () => {
    if (!predictions) return {};
    
    const data = [
      { name: '屈服强度', value: predictions.yield_strength },
      { name: '抗拉强度', value: predictions.tensile_strength },
      { name: '伸长率', value: predictions.elongation },
      { name: '韧性', value: predictions.toughness }
    ];
    
    return {
      title: {
        text: '预测结果',
        left: 'center'
      },
      tooltip: {
        trigger: 'axis',
        axisPointer: {
          type: 'shadow'
        }
      },
      grid: {
        left: '3%',
        right: '4%',
        bottom: '3%',
        containLabel: true
      },
      xAxis: {
        type: 'value'
      },
      yAxis: {
        type: 'category',
        data: data.map(item => item.name)
      },
      series: [
        {
          name: '预测值',
          type: 'bar',
          data: data.map(item => item.value),
          itemStyle: {
            color: function(params) {
              // Different colors for different properties
              const colors = ['#1890ff', '#52c41a', '#faad14', '#722ed1'];
              return colors[params.dataIndex];
            }
          },
          label: {
            show: true,
            position: 'right',
            formatter: '{c}'
          }
        }
      ]
    };
  };

  // 处理批量预测
  const handleBatchPredict = async () => {
    if (!batchData || batchData.length === 0) {
      message.error('请先上传数据');
      return;
    }

    setBatchPredicting(true);
    
    try {
      // 模拟批量预测的API调用
      const results = [];
      
      // 为每条记录生成预测结果
      for (const record of batchData) {
        // 延迟以避免浏览器冻结
        await new Promise(resolve => setTimeout(resolve, 50));
        
        // 模拟预测结果
        const result = {
          ...record, // 保留原始数据
          yield_strength_pred: Math.round(400 + 100 * Math.random()),
          tensile_strength_pred: Math.round(550 + 100 * Math.random()),
          elongation_pred: (20 + 10 * Math.random()).toFixed(1),
          toughness_pred: Math.round(30 + 20 * Math.random())
        };
        
        results.push(result);
      }
      
      setBatchResults(results);
      message.success(`批量预测完成，共处理 ${results.length} 条记录`);
      setActiveTab('results');
    } catch (error) {
      message.error('批量预测失败：' + error.message);
    } finally {
      setBatchPredicting(false);
    }
  };

  // 处理文件上传
  const handleFileUploaded = (data) => {
    if (!data || data.length === 0) {
      setBatchMode(false);
      setBatchData([]);
      return;
    }
    
    setBatchData(data);
    setBatchMode(true);
    message.success(`已加载 ${data.length} 条数据，可以进行批量预测`);
  };

  // 获取批量预测结果表格的列定义
  const getBatchResultColumns = () => {
    if (!batchResults || batchResults.length === 0) {
      return [];
    }
    
    // 从第一行数据中获取所有字段
    const sample = batchResults[0];
    const allFields = Object.keys(sample);
    
    // 分离输入特征和预测结果
    const predictionFields = allFields.filter(field => field.includes('_pred'));
    const inputFields = allFields.filter(field => !field.includes('_pred'));
    
    // 创建输入特征列
    const inputColumns = inputFields.map(field => ({
      title: field,
      dataIndex: field,
      key: field,
      width: 100,
    }));
    
    // 创建预测结果列
    const predictionColumns = predictionFields.map(field => {
      const baseName = field.replace('_pred', '');
      let title = baseName;
      
      // 尝试添加更友好的标题
      if (baseName === 'yield_strength') title = '屈服强度';
      else if (baseName === 'tensile_strength') title = '抗拉强度';
      else if (baseName === 'elongation') title = '伸长率';
      else if (baseName === 'toughness') title = '韧性';
      
      return {
        title: `${title} (预测)`,
        dataIndex: field,
        key: field,
        width: 120,
      };
    });
    
    // 返回组合列，先显示特征后显示预测结果
    return [...inputColumns, ...predictionColumns];
  };

  // 下载预测结果为CSV
  const downloadResults = () => {
    if (!batchResults || batchResults.length === 0) {
      message.error('没有可下载的结果');
      return;
    }
    
    try {
      // 创建CSV内容
      const headers = Object.keys(batchResults[0]).join(',');
      const rows = batchResults.map(row => 
        Object.values(row).map(value => 
          typeof value === 'string' && value.includes(',') ? `"${value}"` : value
        ).join(',')
      );
      const csvContent = [headers, ...rows].join('\n');
      
      // 创建下载链接
      const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.setAttribute('href', url);
      link.setAttribute('download', `钢材性能预测结果_${new Date().toISOString().slice(0, 10)}.csv`);
      link.style.visibility = 'hidden';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      
      message.success('预测结果已下载');
    } catch (error) {
      message.error('下载失败：' + error.message);
    }
  };

  // 获取选项卡配置
  const getTabItems = () => {
    return [
      {
        key: 'input',
        label: '输入参数',
        children: (
          <Row gutter={24}>
            <Col span={16}>
              <Card title="参数输入" variant="outlined">
                {/* 添加单条预测/批量预测切换 */}
                <div style={{ marginBottom: 16 }}>
                  <Space>
                    <Button 
                      type={!batchMode ? 'primary' : 'default'} 
                      onClick={() => setBatchMode(false)}
                    >
                      单条预测
                    </Button>
                    <Button 
                      type={batchMode ? 'primary' : 'default'} 
                      onClick={() => setBatchMode(true)}
                    >
                      批量预测
                    </Button>
                  </Space>
                </div>
                
                {batchMode ? (
                  <div>
                    <FileUploader
                      onFileUploaded={handleFileUploaded}
                      title="上传预测数据"
                      description="请上传符合格式要求的CSV文件，包含所有必需的参数"
                    />
                    
                    <div style={{ marginTop: 16, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    {batchData.length > 0 && (
                        <Text>已加载 {batchData.length} 条记录</Text>
                      )}
                        <Button 
                          type="primary" 
                          onClick={handleBatchPredict}
                          loading={batchPredicting}
                        disabled={!batchData.length}
                        >
                          开始批量预测
                        </Button>
                      </div>
                  </div>
                ) : (
                  <Form
                    form={form}
                    name="prediction_form"
                    layout="vertical"
                    onFinish={handleSubmit}
                    initialValues={{
                      carbon: 0.12,
                      silicon: 1.2,
                      manganese: 0.8,
                      exitTemperature: 950,
                      thickness: 20,
                      coolingTemperature: 600
                    }}
                  >
                    <Row gutter={16}>
                      <Col span={8}>
                        <Form.Item
                          name="carbon"
                          label={
                            <span>
                              碳含量 (C) 
                              <Tooltip title="碳含量范围: 0.08-0.20">
                                <InfoCircleOutlined style={{ marginLeft: 4 }} />
                              </Tooltip>
                            </span>
                          }
                          rules={[{ required: true, message: '请输入碳含量' }]}
                        >
                          <InputNumber
                            min={0.05}
                            max={0.25}
                            step={0.01}
                            style={{ width: '100%' }}
                          />
                        </Form.Item>
                      </Col>
                      <Col span={8}>
                        <Form.Item
                          name="silicon"
                          label={
                            <span>
                              硅含量 (Si)
                              <Tooltip title="硅含量范围: 0.8-1.5">
                                <InfoCircleOutlined style={{ marginLeft: 4 }} />
                              </Tooltip>
                            </span>
                          }
                          rules={[{ required: true, message: '请输入硅含量' }]}
                        >
                          <InputNumber
                            min={0.5}
                            max={2.0}
                            step={0.1}
                            style={{ width: '100%' }}
                          />
                        </Form.Item>
                      </Col>
                      <Col span={8}>
                        <Form.Item
                          name="manganese"
                          label={
                            <span>
                              锰含量 (Mn)
                              <Tooltip title="锰含量范围: 0.3-1.5">
                                <InfoCircleOutlined style={{ marginLeft: 4 }} />
                              </Tooltip>
                            </span>
                          }
                          rules={[{ required: true, message: '请输入锰含量' }]}
                        >
                          <InputNumber
                            min={0.1}
                            max={2.0}
                            step={0.1}
                            style={{ width: '100%' }}
                          />
                        </Form.Item>
                      </Col>
                    </Row>

                    <Row gutter={16}>
                      <Col span={8}>
                        <Form.Item
                          name="exitTemperature"
                          label={
                            <span>
                              出炉温度 (°C)
                              <Tooltip title="出炉温度范围: 900-1000°C">
                                <InfoCircleOutlined style={{ marginLeft: 4 }} />
                              </Tooltip>
                            </span>
                          }
                          rules={[{ required: true, message: '请输入出炉温度' }]}
                        >
                          <InputNumber
                            min={800}
                            max={1200}
                            style={{ width: '100%' }}
                          />
                        </Form.Item>
                      </Col>
                      <Col span={8}>
                        <Form.Item
                          name="thickness"
                          label={
                            <span>
                              钢板厚度 (mm)
                              <Tooltip title="钢板厚度范围: 5-40mm">
                                <InfoCircleOutlined style={{ marginLeft: 4 }} />
                              </Tooltip>
                            </span>
                          }
                          rules={[{ required: true, message: '请输入钢板厚度' }]}
                        >
                          <InputNumber
                            min={5}
                            max={40}
                            style={{ width: '100%' }}
                          />
                        </Form.Item>
                      </Col>
                      <Col span={8}>
                        <Form.Item
                          name="coolingTemperature"
                          label={
                            <span>
                              水冷温度 (°C)
                              <Tooltip title="水冷温度范围: 550-650°C">
                                <InfoCircleOutlined style={{ marginLeft: 4 }} />
                              </Tooltip>
                            </span>
                          }
                          rules={[{ required: true, message: '请输入水冷温度' }]}
                        >
                          <InputNumber
                            min={500}
                            max={700}
                            style={{ width: '100%' }}
                          />
                        </Form.Item>
                      </Col>
                    </Row>

                    <Form.Item>
                      <Button type="primary" htmlType="submit" loading={loading}>
                        开始预测
                      </Button>
                    </Form.Item>
                  </Form>
                )}
              </Card>
            </Col>
            
            {/* 右侧说明卡片 */}
            <Col span={8}>
              <Card title="使用说明" variant="outlined">
                <Typography.Paragraph>
                  <Typography.Text strong>单条预测：</Typography.Text> 输入单个钢材样本的化学成分和工艺参数，获取预测的机械性能。
                </Typography.Paragraph>
                <Typography.Paragraph>
                  <Typography.Text strong>批量预测：</Typography.Text> 上传CSV文件进行批量预测，CSV文件需包含必要的成分和工艺参数。
                </Typography.Paragraph>
                <Typography.Paragraph>
                  <Typography.Text type="secondary">
                    预测模型基于大量历史数据训练，支持多种材质钢材的性能预测。
                  </Typography.Text>
                </Typography.Paragraph>
              </Card>
            </Col>
          </Row>
        )
      },
      {
        key: 'results',
        label: '预测结果',
        children: (
          <Spin spinning={loading || batchPredicting}>
            {batchMode ? (
              // 批量预测结果展示
              <Card title="批量预测结果">
                {batchResults.length > 0 ? (
                  <div>
                    <div style={{ marginBottom: 16, display: 'flex', justifyContent: 'space-between' }}>
                      <Typography.Text strong>共 {batchResults.length} 条预测结果</Typography.Text>
                      <Button 
                        type="primary" 
                        onClick={downloadResults}
                        icon={<FileOutlined />}
                      >
                        导出结果
                      </Button>
                    </div>
                    <Table 
                      dataSource={batchResults} 
                      columns={getBatchResultColumns()}
                      scroll={{ x: 'max-content' }}
                      pagination={{ pageSize: 10 }}
                      rowKey={(record) => record.id || Math.random().toString(36).substr(2, 9)}
                      size="small"
                    />
                  </div>
                ) : (
                  <Typography.Text>请先进行批量预测</Typography.Text>
                )}
              </Card>
            ) : (
              // 单条预测结果展示
              predictions ? (
                <Card title="预测结果">
                  <Row gutter={[24, 24]}>
                    <Col span={12}>
                      <Card type="inner" title="屈服强度">
                        <Typography.Title level={3} style={{ color: '#1890ff' }}>
                          {predictions.yield_strength.toFixed(1)} MPa
                        </Typography.Title>
                      </Card>
                    </Col>
                    <Col span={12}>
                      <Card type="inner" title="抗拉强度">
                        <Typography.Title level={3} style={{ color: '#52c41a' }}>
                          {predictions.tensile_strength.toFixed(1)} MPa
                        </Typography.Title>
                      </Card>
                    </Col>
                    <Col span={12}>
                      <Card type="inner" title="伸长率">
                        <Typography.Title level={3} style={{ color: '#faad14' }}>
                          {predictions.elongation.toFixed(1)} %
                        </Typography.Title>
                      </Card>
                    </Col>
                    <Col span={12}>
                      <Card type="inner" title="冲击韧性">
                        <Typography.Title level={3} style={{ color: '#722ed1' }}>
                          {predictions.toughness.toFixed(1)} J
                        </Typography.Title>
                      </Card>
                    </Col>
                  </Row>

                  <div style={{ marginTop: 24, textAlign: 'center' }}>
                    <Button 
                      onClick={() => setActiveTab('input')} 
                      type="primary"
                    >
                      返回修改参数
                    </Button>
                  </div>
                </Card>
              ) : (
                <Card>
                  <Typography.Text>请先进行预测</Typography.Text>
                </Card>
              )
            )}
          </Spin>
        )
      }
    ];
  };

  return (
    <div className="prediction-page">
      <Typography.Title level={2}>钢材性能预测</Typography.Title>
      <Typography.Paragraph>
        基于钢材的化学成分和工艺参数，预测最终的机械性能，包括屈服强度、抗拉强度、伸长率和冲击韧性。
      </Typography.Paragraph>
      
      <Tabs activeKey={activeTab} onChange={setActiveTab} items={getTabItems()} />
    </div>
  );
};

export default PredictionForm; 