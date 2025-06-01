const React = require('react');
const { render, screen, fireEvent, waitFor } = require('@testing-library/react');
const { act } = require('react-dom/test-utils');
const DataAnalysis = require('../../src/pages/DataAnalysis').default;

// 模拟echarts-for-react组件
jest.mock('echarts-for-react', () => {
  return function DummyECharts(props) {
    return <div data-testid="mock-echarts" data-options={JSON.stringify(props.option)} />;
  };
});

// 模拟FileUploader组件
jest.mock('../../src/components/FileUploader', () => {
  return function MockFileUploader({ onFileUploaded }) {
    return (
      <div>
        <button 
          data-testid="mock-upload-button"
          onClick={() => onFileUploaded([
            { c: 0.12, si: 1.2, tensile_strength: 650 },
            { c: 0.15, si: 1.1, tensile_strength: 670 }
          ])}
        >
          上传文件
        </button>
      </div>
    );
  };
});

// 模拟message
jest.mock('antd', () => {
  const antd = jest.requireActual('antd');
  return {
    ...antd,
    message: {
      success: jest.fn(),
      error: jest.fn(),
    },
  };
});

describe('DataAnalysis Component', () => {
  beforeEach(() => {
    // 清除所有模拟调用
    jest.clearAllMocks();
  });

  test('renders loading state initially', () => {
    render(<DataAnalysis />);
    
    // 检查加载状态
    expect(screen.getByText(/加载中/i)).toBeInTheDocument();
  });

  test('renders data analysis tabs after loading', async () => {
    render(<DataAnalysis />);
    
    // 等待加载完成
    await waitFor(() => {
      expect(screen.queryByText(/加载中/i)).not.toBeInTheDocument();
    });
    
    // 检查主要组件是否已渲染
    expect(screen.getByText(/趋势分析/i)).toBeInTheDocument();
    expect(screen.getByText(/相关性分析/i)).toBeInTheDocument();
    expect(screen.getByText(/数据表/i)).toBeInTheDocument();
  });

  test('allows user to switch between tabs', async () => {
    render(<DataAnalysis />);
    
    // 等待加载完成
    await waitFor(() => {
      expect(screen.queryByText(/加载中/i)).not.toBeInTheDocument();
    });
    
    // 点击相关性分析标签
    const correlationTab = screen.getByText(/相关性分析/i);
    fireEvent.click(correlationTab);
    
    // 假设相关性分析标签页有独特的文字或元素
    await waitFor(() => {
      expect(screen.getByTestId('mock-echarts')).toBeInTheDocument();
    });
  });

  test('allows selecting different properties', async () => {
    render(<DataAnalysis />);
    
    // 等待加载完成
    await waitFor(() => {
      expect(screen.queryByText(/加载中/i)).not.toBeInTheDocument();
    });
    
    // 假设有一个属性选择器
    const propertySelect = screen.getByLabelText(/选择属性/i) || screen.getByText(/屈服强度/i);
    fireEvent.mouseDown(propertySelect);
    
    // 等待下拉选项出现
    await waitFor(() => {
      const tensileOption = screen.getByText(/抗拉强度/i);
      fireEvent.click(tensileOption);
    });
    
    // 验证图表已更新
    expect(screen.getByTestId('mock-echarts')).toBeInTheDocument();
  });

  test('handles custom data upload', async () => {
    const { message } = require('antd');
    render(<DataAnalysis />);
    
    // 等待加载完成
    await waitFor(() => {
      expect(screen.queryByText(/加载中/i)).not.toBeInTheDocument();
    });
    
    // 点击上传按钮
    const uploadButton = screen.getByTestId('mock-upload-button');
    fireEvent.click(uploadButton);
    
    // 验证成功消息已显示
    expect(message.success).toHaveBeenCalledWith('数据加载成功，可以开始分析');
    
    // 验证可用的因子和属性已更新
    await waitFor(() => {
      expect(screen.getByText(/c/i)).toBeInTheDocument();
      expect(screen.getByText(/si/i)).toBeInTheDocument();
      expect(screen.getByText(/tensile_strength/i)).toBeInTheDocument();
    });
  });
}); 