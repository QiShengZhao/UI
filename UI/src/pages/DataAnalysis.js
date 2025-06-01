import React, { useState, useEffect } from 'react';
import { 
  Card, 
  Row, 
  Col, 
  Typography, 
  Select, 
  Tabs,
  Button,
  Spin,
  Slider,
  Space,
  Divider,
  Table,
  Tooltip,
  message,
  Alert
} from 'antd';
import {
  LineChartOutlined,
  BarChartOutlined,
  HeatMapOutlined,
  DotChartOutlined,
  QuestionCircleOutlined
} from '@ant-design/icons';
import ReactECharts from 'echarts-for-react';
import FileUploader from '../components/FileUploader';

const { Title, Text, Paragraph } = Typography;
const { Option } = Select;
const { OptGroup } = Select;
const { TabPane } = Tabs;

// 样本数据 - 在真实应用中应该从API获取
const sampleCompositionData = [
  { c: 0.12, si: 1.2, mn: 0.8, p: 0.02, s: 0.01, cr: 0.01, ni: 0.02, yield_strength: 450, tensile_strength: 650, elongation: 22, toughness: 45 },
  { c: 0.15, si: 1.1, mn: 0.9, p: 0.02, s: 0.01, cr: 0.01, ni: 0.02, yield_strength: 480, tensile_strength: 670, elongation: 20, toughness: 42 },
  { c: 0.11, si: 1.3, mn: 0.7, p: 0.02, s: 0.01, cr: 0.01, ni: 0.02, yield_strength: 430, tensile_strength: 640, elongation: 24, toughness: 48 },
  { c: 0.13, si: 1.0, mn: 1.0, p: 0.02, s: 0.01, cr: 0.01, ni: 0.02, yield_strength: 460, tensile_strength: 660, elongation: 21, toughness: 44 },
  { c: 0.14, si: 1.2, mn: 0.8, p: 0.02, s: 0.01, cr: 0.01, ni: 0.02, yield_strength: 470, tensile_strength: 665, elongation: 21, toughness: 43 },
  { c: 0.10, si: 1.1, mn: 0.9, p: 0.02, s: 0.01, cr: 0.01, ni: 0.02, yield_strength: 420, tensile_strength: 630, elongation: 25, toughness: 49 },
  { c: 0.16, si: 1.0, mn: 0.7, p: 0.02, s: 0.01, cr: 0.01, ni: 0.02, yield_strength: 490, tensile_strength: 680, elongation: 19, toughness: 40 },
  { c: 0.13, si: 1.3, mn: 0.8, p: 0.02, s: 0.01, cr: 0.01, ni: 0.02, yield_strength: 455, tensile_strength: 655, elongation: 22, toughness: 45 },
  { c: 0.12, si: 1.2, mn: 1.0, p: 0.02, s: 0.01, cr: 0.01, ni: 0.02, yield_strength: 465, tensile_strength: 665, elongation: 21, toughness: 44 },
  { c: 0.14, si: 1.1, mn: 0.8, p: 0.02, s: 0.01, cr: 0.01, ni: 0.02, yield_strength: 475, tensile_strength: 670, elongation: 20, toughness: 42 },
];

const sampleProcessData = [
  { temp: 950, thickness: 15, coolTemp: 600, yield_strength: 450, tensile_strength: 650, elongation: 22, toughness: 45 },
  { temp: 930, thickness: 12, coolTemp: 580, yield_strength: 480, tensile_strength: 670, elongation: 20, toughness: 42 },
  { temp: 940, thickness: 18, coolTemp: 610, yield_strength: 430, tensile_strength: 640, elongation: 24, toughness: 48 },
  { temp: 960, thickness: 14, coolTemp: 590, yield_strength: 460, tensile_strength: 660, elongation: 21, toughness: 44 },
  { temp: 950, thickness: 16, coolTemp: 600, yield_strength: 470, tensile_strength: 665, elongation: 21, toughness: 43 },
  { temp: 930, thickness: 20, coolTemp: 620, yield_strength: 420, tensile_strength: 630, elongation: 25, toughness: 49 },
  { temp: 970, thickness: 10, coolTemp: 570, yield_strength: 490, tensile_strength: 680, elongation: 19, toughness: 40 },
  { temp: 945, thickness: 15, coolTemp: 595, yield_strength: 455, tensile_strength: 655, elongation: 22, toughness: 45 },
  { temp: 955, thickness: 17, coolTemp: 605, yield_strength: 465, tensile_strength: 665, elongation: 21, toughness: 44 },
  { temp: 935, thickness: 13, coolTemp: 585, yield_strength: 475, tensile_strength: 670, elongation: 20, toughness: 42 },
];

const DataAnalysis = () => {
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('trend');
  const [selectedProperty, setSelectedProperty] = useState('yield_strength');
  const [selectedFactor, setSelectedFactor] = useState('c');
  const [compositionData, setCompositionData] = useState([]);
  const [processData, setProcessData] = useState([]);
  const [dataRange, setDataRange] = useState([0, 100]);
  const [useCustomData, setUseCustomData] = useState(false);
  const [customData, setCustomData] = useState([]);
  const [availableFactors, setAvailableFactors] = useState([]);
  const [availableProperties, setAvailableProperties] = useState([]);

  // 用于模拟数据加载
  useEffect(() => {
    setTimeout(() => {
      setCompositionData(sampleCompositionData);
      setProcessData(sampleProcessData);
      setLoading(false);
    }, 1000);
  }, []);

  // 处理上传的CSV数据
  const handleFileUploaded = (data) => {
    if (!data || data.length === 0) {
      setUseCustomData(false);
      return;
    }
    
    // 验证数据结构
    try {
      // 检查数据中的列名
      const sampleRow = data[0];
      const fields = Object.keys(sampleRow);
      
      // 检测可用的因子和属性
      const detectedFactors = [];
      const detectedProperties = [];
      
      // 预定义的钢材性能属性
      const knownProperties = [
        'yield_strength', 'tensile_strength', 'elongation', 'toughness', 
        '屈服强度', '抗拉强度', '伸长率', '韧性', '实测1', '实测2', '实测3'
      ];
      
      // 遍历字段，识别因子和属性
      fields.forEach(field => {
        // 尝试规范化字段名（小写，删除空格）
        const normalizedField = field.toLowerCase().replace(/\s+/g, '_');
        
        // 检查是否是已知的性能属性
        if (knownProperties.includes(normalizedField) || knownProperties.includes(field)) {
          detectedProperties.push(field);
        } else {
          detectedFactors.push(field);
        }
      });
      
      if (detectedFactors.length === 0 || detectedProperties.length === 0) {
        message.error('无法识别数据中的因子和属性列，请检查CSV文件格式');
        return;
      }
      
      // 更新可用因子和属性
      setAvailableFactors(detectedFactors);
      setAvailableProperties(detectedProperties);
      
      // 设置默认选中的因子和属性
      setSelectedFactor(detectedFactors[0]);
      setSelectedProperty(detectedProperties[0]);
      
      // 更新自定义数据
      setCustomData(data);
      setUseCustomData(true);
      message.success('数据加载成功，可以开始分析');
      
    } catch (error) {
      message.error('数据解析失败: ' + error.message);
    }
  };

  // 不同属性的中文名映射
  const propertyLabels = {
    yield_strength: '屈服强度',
    tensile_strength: '抗拉强度',
    elongation: '伸长率',
    toughness: '韧性'
  };

  const factorLabels = {
    // 成分因素
    c: '碳含量 (C)',
    si: '硅含量 (Si)',
    mn: '锰含量 (Mn)',
    p: '磷含量 (P)',
    s: '硫含量 (S)',
    cr: '铬含量 (Cr)',
    ni: '镍含量 (Ni)',
    // 工艺因素
    temp: '出炉温度',
    thickness: '钢板厚度',
    coolTemp: '水冷温度'
  };

  // 获取当前使用的数据集
  const getCurrentDataSet = () => {
    if (useCustomData && customData.length > 0) {
      return customData;
    }
    
    return ['c', 'si', 'mn', 'p', 's', 'cr', 'ni'].includes(selectedFactor) 
      ? compositionData 
      : processData;
  };

  // 获取当前因子的标签名
  const getFactorLabel = (factor) => {
    if (useCustomData) {
      return factorLabels[factor] || factor; // 如果没有预定义标签就使用原字段名
    }
    return factorLabels[factor];
  };

  // 获取当前属性的标签名
  const getPropertyLabel = (property) => {
    if (useCustomData) {
      return propertyLabels[property] || property; // 如果没有预定义标签就使用原字段名
    }
    return propertyLabels[property];
  };

  // 趋势分析图表选项
  const getTrendOption = () => {
    // 获取当前数据集
    const dataSet = getCurrentDataSet();
    
    // 排序数据以便图表平滑
    const sortedData = [...dataSet].sort((a, b) => {
      const factorKey = selectedFactor;
      return a[factorKey] - b[factorKey];
    });
    
    // 提取横坐标和纵坐标数据
    const xData = sortedData.map(item => item[selectedFactor]);
    const yData = sortedData.map(item => item[selectedProperty]);
    
    return {
      title: {
        text: `${getFactorLabel(selectedFactor)}对${getPropertyLabel(selectedProperty)}的影响`,
        left: 'center'
      },
      tooltip: {
        trigger: 'axis',
        formatter: function(params) {
          const dataIndex = params[0].dataIndex;
          const item = sortedData[dataIndex];
          return `${getFactorLabel(selectedFactor)}: ${item[selectedFactor]}<br/>` +
                 `${getPropertyLabel(selectedProperty)}: ${item[selectedProperty]}`;
        }
      },
      xAxis: {
        type: 'category',
        name: getFactorLabel(selectedFactor),
        data: xData,
        nameLocation: 'middle',
        nameGap: 30
      },
      yAxis: {
        type: 'value',
        name: getPropertyLabel(selectedProperty),
        nameLocation: 'middle',
        nameGap: 50
      },
      series: [
        {
          name: getPropertyLabel(selectedProperty),
          type: 'line',
          data: yData,
          smooth: true,
          markPoint: {
            data: [
              { type: 'max', name: '最大值' },
              { type: 'min', name: '最小值' }
            ]
          },
          markLine: {
            data: [
              { type: 'average', name: '平均值' }
            ]
          }
        }
      ]
    };
  };

  // 相关性分析图表选项
  const getCorrelationOption = () => {
    // 使用成分数据集进行相关性分析
    const dataSet = compositionData;
    
    // 计算相关性矩阵
    const factors = ['c', 'si', 'mn', 'p', 's', 'cr', 'ni'];
    const properties = ['yield_strength', 'tensile_strength', 'elongation', 'toughness'];
    
    // 模拟相关性系数 (在实际应用中应该计算真实的相关性)
    const correlationData = [
      // C, Si, Mn, P, S, Cr, Ni
      [0.85, 0.65, 0.70, -0.40, -0.50, 0.30, 0.25], // 屈服强度
      [0.80, 0.60, 0.75, -0.45, -0.55, 0.35, 0.20], // 抗拉强度
      [-0.70, -0.40, -0.50, 0.30, 0.40, -0.20, -0.15], // 伸长率
      [-0.60, -0.30, -0.45, 0.35, 0.45, -0.25, -0.10]  // 韧性
    ];
    
    return {
      title: {
        text: '钢材成分与性能相关性分析',
        left: 'center'
      },
      tooltip: {
        position: 'top',
        formatter: function (params) {
          return `${factorLabels[factors[params.data[1]]]}<br/>${propertyLabels[properties[params.data[0]]]}<br/>相关性: ${params.data[2].toFixed(2)}`;
        }
      },
      grid: {
        left: '15%',
        right: '10%',
        bottom: '15%',
        top: '15%'
      },
      xAxis: {
        type: 'category',
        data: factors.map(f => factorLabels[f]),
        splitArea: {
          show: true
        },
        axisLabel: {
          interval: 0,
          rotate: 30
        }
      },
      yAxis: {
        type: 'category',
        data: properties.map(p => propertyLabels[p]),
        splitArea: {
          show: true
        }
      },
      visualMap: {
        min: -1,
        max: 1,
        calculable: true,
        orient: 'horizontal',
        left: 'center',
        bottom: '0%',
        inRange: {
          color: ['#d73027', '#f46d43', '#fdae61', '#fee090', '#ffffbf', '#e0f3f8', '#abd9e9', '#74add1', '#4575b4']
        }
      },
      series: [{
        name: '相关性',
        type: 'heatmap',
        data: correlationData.flatMap((row, i) => 
          row.map((val, j) => [i, j, val])
        ),
        emphasis: {
          itemStyle: {
            shadowBlur: 10,
            shadowColor: 'rgba(0, 0, 0, 0.5)'
          }
        }
      }]
    };
  };

  // 散点分布图表选项
  const getScatterOption = () => {
    // 确定使用哪个数据集
    const dataSet = ['c', 'si', 'mn', 'p', 's', 'cr', 'ni'].includes(selectedFactor) 
      ? compositionData 
      : processData;
    
    // 提取散点数据
    const scatterData = dataSet.map(item => [item[selectedFactor], item[selectedProperty]]);
    
    return {
      title: {
        text: `${factorLabels[selectedFactor]}与${propertyLabels[selectedProperty]}的关系`,
        left: 'center'
      },
      tooltip: {
        trigger: 'item',
        formatter: function(params) {
          return `${factorLabels[selectedFactor]}: ${params.value[0]}<br/>${propertyLabels[selectedProperty]}: ${params.value[1]}`;
        }
      },
      xAxis: {
        type: 'value',
        name: factorLabels[selectedFactor],
        nameLocation: 'middle',
        nameGap: 30
      },
      yAxis: {
        type: 'value',
        name: propertyLabels[selectedProperty],
        nameLocation: 'middle',
        nameGap: 50
      },
      series: [
        {
          name: '数据点',
          type: 'scatter',
          data: scatterData,
          symbolSize: 12,
          emphasis: {
            itemStyle: {
              shadowBlur: 10,
              shadowColor: 'rgba(0, 0, 0, 0.5)'
            }
          }
        },
        {
          name: '拟合线',
          type: 'line',
          smooth: true,
          showSymbol: false,
          data: calculateRegressionLine(scatterData),
          itemStyle: {
            color: '#1890ff'
          }
        }
      ]
    };
  };

  // 计算简单线性回归线
  const calculateRegressionLine = (data) => {
    if (data.length < 2) return [];
    
    // 提取x和y值
    const xValues = data.map(point => point[0]);
    const yValues = data.map(point => point[1]);
    
    // 计算均值
    const xMean = xValues.reduce((sum, val) => sum + val, 0) / xValues.length;
    const yMean = yValues.reduce((sum, val) => sum + val, 0) / yValues.length;
    
    // 计算斜率和截距
    let numerator = 0;
    let denominator = 0;
    
    for (let i = 0; i < data.length; i++) {
      numerator += (xValues[i] - xMean) * (yValues[i] - yMean);
      denominator += Math.pow(xValues[i] - xMean, 2);
    }
    
    const slope = denominator !== 0 ? numerator / denominator : 0;
    const intercept = yMean - slope * xMean;
    
    // 生成回归线点
    const minX = Math.min(...xValues);
    const maxX = Math.max(...xValues);
    
    return [
      [minX, slope * minX + intercept],
      [maxX, slope * maxX + intercept]
    ];
  };

  // 数据表格列定义
  const getColumns = () => {
    const isComposition = ['c', 'si', 'mn', 'p', 's', 'cr', 'ni'].includes(selectedFactor);
    
    if (isComposition) {
      return [
        { title: '碳含量 (C)', dataIndex: 'c', key: 'c', sorter: (a, b) => a.c - b.c },
        { title: '硅含量 (Si)', dataIndex: 'si', key: 'si', sorter: (a, b) => a.si - b.si },
        { title: '锰含量 (Mn)', dataIndex: 'mn', key: 'mn', sorter: (a, b) => a.mn - b.mn },
        { title: '屈服强度', dataIndex: 'yield_strength', key: 'yield_strength', sorter: (a, b) => a.yield_strength - b.yield_strength },
        { title: '抗拉强度', dataIndex: 'tensile_strength', key: 'tensile_strength', sorter: (a, b) => a.tensile_strength - b.tensile_strength },
        { title: '伸长率 (%)', dataIndex: 'elongation', key: 'elongation', sorter: (a, b) => a.elongation - b.elongation },
        { title: '韧性 (J)', dataIndex: 'toughness', key: 'toughness', sorter: (a, b) => a.toughness - b.toughness }
      ];
    } else {
      return [
        { title: '出炉温度 (°C)', dataIndex: 'temp', key: 'temp', sorter: (a, b) => a.temp - b.temp },
        { title: '钢板厚度 (mm)', dataIndex: 'thickness', key: 'thickness', sorter: (a, b) => a.thickness - b.thickness },
        { title: '水冷温度 (°C)', dataIndex: 'coolTemp', key: 'coolTemp', sorter: (a, b) => a.coolTemp - b.coolTemp },
        { title: '屈服强度', dataIndex: 'yield_strength', key: 'yield_strength', sorter: (a, b) => a.yield_strength - b.yield_strength },
        { title: '抗拉强度', dataIndex: 'tensile_strength', key: 'tensile_strength', sorter: (a, b) => a.tensile_strength - b.tensile_strength },
        { title: '伸长率 (%)', dataIndex: 'elongation', key: 'elongation', sorter: (a, b) => a.elongation - b.elongation },
        { title: '韧性 (J)', dataIndex: 'toughness', key: 'toughness', sorter: (a, b) => a.toughness - b.toughness }
      ];
    }
  };

  // 修改选择属性的处理函数
  const handlePropertyChange = (value) => {
    setSelectedProperty(value);
  };

  // 修改选择因子的处理函数
  const handleFactorChange = (value) => {
    setSelectedFactor(value);
  };

  // 渲染属性选择器选项
  const renderPropertyOptions = () => {
    if (useCustomData) {
      return availableProperties.map(prop => (
        <Option key={prop} value={prop}>{propertyLabels[prop] || prop}</Option>
      ));
    }
    
    return Object.entries(propertyLabels).map(([key, label]) => (
      <Option key={key} value={key}>{label}</Option>
    ));
  };

  // 渲染因子选择器选项
  const renderFactorOptions = () => {
    if (useCustomData) {
      return availableFactors.map(factor => (
        <Option key={factor} value={factor}>{factorLabels[factor] || factor}</Option>
      ));
    }
    
    return (
      <>
        <OptGroup label="成分因素">
          <Option value="c">碳含量 (C)</Option>
          <Option value="si">硅含量 (Si)</Option>
          <Option value="mn">锰含量 (Mn)</Option>
          <Option value="p">磷含量 (P)</Option>
          <Option value="s">硫含量 (S)</Option>
          <Option value="cr">铬含量 (Cr)</Option>
          <Option value="ni">镍含量 (Ni)</Option>
        </OptGroup>
        <OptGroup label="工艺因素">
          <Option value="temp">出炉温度</Option>
          <Option value="thickness">钢板厚度</Option>
          <Option value="coolTemp">水冷温度</Option>
        </OptGroup>
      </>
    );
  };

  // 下载报告
  const handleDownloadReport = () => {
    message.success('分析报告已开始下载');
    // 这里应该是真实的报告下载逻辑
  };

  if (loading) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '80vh' }}>
        <Spin size="large" />
      </div>
    );
  }

  // Tabs items configuration
  const tabItems = [
    {
      key: 'trend',
      label: <span><LineChartOutlined /> 趋势分析</span>,
      children: (
        <>
          <ReactECharts 
            option={getTrendOption()} 
            style={{ height: '400px' }} 
            opts={{ renderer: 'svg' }}
          />
          
          <Divider />
          
          <Paragraph>
            <Text strong>分析说明: </Text>
            上图展示了{factorLabels[selectedFactor]}对{propertyLabels[selectedProperty]}的影响趋势。
            可以看出，随着{factorLabels[selectedFactor]}的变化，{propertyLabels[selectedProperty]}
            {selectedProperty === 'yield_strength' || selectedProperty === 'tensile_strength' ? 
              (selectedFactor === 'c' ? '呈现上升趋势' : selectedFactor === 'mn' ? '有所提高' : '波动不大') : 
              (selectedFactor === 'c' ? '呈现下降趋势' : selectedFactor === 'mn' ? '略有下降' : '变化不明显')
            }。
            <Tooltip title="点击查看更多关于该因素的详细信息">
              <QuestionCircleOutlined style={{ marginLeft: 8 }} />
            </Tooltip>
          </Paragraph>
        </>
      )
    },
    {
      key: 'correlation',
      label: <span><HeatMapOutlined /> 相关性分析</span>,
      children: (
        <>
          <ReactECharts 
            option={getCorrelationOption()} 
            style={{ height: '400px' }} 
            opts={{ renderer: 'svg' }}
          />
          
          <Divider />
          
          <Paragraph>
            <Text strong>相关性分析说明: </Text>
            热力图显示了不同钢材成分与性能指标之间的相关性强度。颜色越深表示相关性越强，
            红色表示正相关(随着成分增加，性能指标上升)，蓝色表示负相关(随着成分增加，性能指标下降)。
            可以看出，碳含量(C)与屈服强度和抗拉强度有较强的正相关性，而与伸长率和韧性呈现负相关。
            <Tooltip title="点击查看相关性系数的计算方法">
              <QuestionCircleOutlined style={{ marginLeft: 8 }} />
            </Tooltip>
          </Paragraph>
        </>
      )
    },
    {
      key: 'scatter',
      label: <span><DotChartOutlined /> 散点分析</span>,
      children: (
        <>
          <ReactECharts 
            option={getScatterOption()} 
            style={{ height: '400px' }} 
            opts={{ renderer: 'svg' }}
          />
          
          <Divider />
          
          <Paragraph>
            <Text strong>散点分析说明: </Text>
            散点图展示了{factorLabels[selectedFactor]}与{propertyLabels[selectedProperty]}之间的分布关系。
            蓝线表示线性回归拟合曲线，可以观察到数据点的分布趋势。
            数据点的分散程度反映了影响因素与性能指标之间关联的稳定性。
            <Tooltip title="点击查看更多关于散点分析的解读方法">
              <QuestionCircleOutlined style={{ marginLeft: 8 }} />
            </Tooltip>
          </Paragraph>
        </>
      )
    },
    {
      key: 'data',
      label: <span><BarChartOutlined /> 数据表</span>,
      children: (
        <Table 
          columns={getColumns()} 
          dataSource={['c', 'si', 'mn', 'p', 's', 'cr', 'ni'].includes(selectedFactor) ? compositionData : processData} 
          rowKey={(record) => record.id || Math.random().toString(36).substr(2, 9)}
          pagination={{ pageSize: 5 }}
          scroll={{ x: 'max-content' }}
        />
      )
    }
  ];

  return (
    <div className="data-analysis-page">
      <Title level={2}>数据分析</Title>
      <Paragraph>
        通过分析钢材成分和工艺参数与性能之间的关系，可以帮助优化生产过程，提高产品质量。
      </Paragraph>
      
      {/* 文件上传器 */}
      <Card title="数据源选择" style={{ marginBottom: 24 }}>
        <FileUploader 
          onFileUploaded={handleFileUploaded}
          title="上传分析数据"
          description="请上传包含钢材成分/工艺参数和性能数据的CSV文件"
        />
        {useCustomData && (
          <Alert
            message="自定义数据已加载"
            description={`共 ${customData.length} 条记录，${availableFactors.length} 个因子，${availableProperties.length} 个性能指标。`}
            type="success"
            showIcon
            style={{ marginTop: 16 }}
          />
        )}
      </Card>
      
      <Card>
        <Row gutter={16} style={{ marginBottom: 16 }}>
          <Col span={12}>
            <Space>
              <Text>选择性能指标:</Text>
              <Select
                style={{ width: 200 }}
                value={selectedProperty}
                onChange={handlePropertyChange}
              >
                {renderPropertyOptions()}
              </Select>
            </Space>
          </Col>
          <Col span={12}>
            <Space>
              <Text>选择影响因子:</Text>
              <Select
                style={{ width: 200 }}
                value={selectedFactor}
                onChange={handleFactorChange}
              >
                {renderFactorOptions()}
              </Select>
            </Space>
          </Col>
        </Row>

        <Tabs activeKey={activeTab} onChange={setActiveTab} items={tabItems} />
      </Card>
    </div>
  );
};

export default DataAnalysis; 