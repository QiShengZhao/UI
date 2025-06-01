const React = require('react');
const { render, screen, fireEvent, waitFor } = require('@testing-library/react');
const { act } = require('react-dom/test-utils');
const PredictionForm = require('../../src/pages/PredictionForm').default;

// 模拟API服务
jest.mock('../../src/services/api', () => ({
  makePrediction: jest.fn(),
}));

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
      <button 
        data-testid="mock-upload-button"
        onClick={() => onFileUploaded([
          { c: 0.12, si: 0.25, mn: 1.4, temp: 950, thickness: 15, coolTemp: 600 },
          { c: 0.15, si: 0.30, mn: 1.2, temp: 930, thickness: 12, coolTemp: 580 }
        ])}
      >
        上传数据
      </button>
    );
  };
});

describe('PredictionForm Component', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('renders form elements', () => {
    render(<PredictionForm />);
    
    // 检查标题是否存在
    expect(screen.getByText(/钢材性能预测/i)).toBeInTheDocument();
    
    // 检查输入表单字段
    expect(screen.getByLabelText(/碳含量/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/硅含量/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/锰含量/i)).toBeInTheDocument();
  });

  test('handles form submission', async () => {
    const ApiService = require('../../src/services/api');
    const mockPredictData = {
      success: true,
      predictions: {
        yield_strength: 450,
        tensile_strength: 650,
        elongation: 22,
        toughness: 45
      }
    };
    
    // 设置API返回值
    ApiService.makePrediction.mockResolvedValue(mockPredictData);
    
    render(<PredictionForm />);
    
    // 填写表单数据
    fireEvent.change(screen.getByLabelText(/碳含量/i), { target: { value: '0.15' } });
    fireEvent.change(screen.getByLabelText(/硅含量/i), { target: { value: '0.25' } });
    fireEvent.change(screen.getByLabelText(/锰含量/i), { target: { value: '1.4' } });
    fireEvent.change(screen.getByLabelText(/出炉温度/i), { target: { value: '950' } });
    fireEvent.change(screen.getByLabelText(/厚度/i), { target: { value: '15' } });
    fireEvent.change(screen.getByLabelText(/水冷温度/i), { target: { value: '600' } });
    
    // 提交表单
    const submitButton = screen.getByText(/开始预测/i);
    fireEvent.click(submitButton);
    
    // 验证API被正确调用
    expect(ApiService.makePrediction).toHaveBeenCalledWith({
      c: '0.15',
      si: '0.25',
      mn: '1.4',
      temp: '950',
      thickness: '15', 
      coolTemp: '600'
    });
    
    // 等待预测结果显示
    await waitFor(() => {
      expect(screen.getByTestId('mock-echarts')).toBeInTheDocument();
    });
  });

  test('handles API errors', async () => {
    const ApiService = require('../../src/services/api');
    
    // 模拟API错误
    ApiService.makePrediction.mockRejectedValue(new Error('服务器错误'));
    
    render(<PredictionForm />);
    
    // 填写表单数据
    fireEvent.change(screen.getByLabelText(/碳含量/i), { target: { value: '0.15' } });
    fireEvent.change(screen.getByLabelText(/硅含量/i), { target: { value: '0.25' } });
    fireEvent.change(screen.getByLabelText(/锰含量/i), { target: { value: '1.4' } });
    
    // 提交表单
    const submitButton = screen.getByText(/开始预测/i);
    fireEvent.click(submitButton);
    
    // 等待错误信息显示
    await waitFor(() => {
      expect(screen.getByText(/服务器错误/i)).toBeInTheDocument();
    });
  });

  test('handles form reset', () => {
    render(<PredictionForm />);
    
    // 填写表单数据
    fireEvent.change(screen.getByLabelText(/碳含量/i), { target: { value: '0.15' } });
    fireEvent.change(screen.getByLabelText(/硅含量/i), { target: { value: '0.25' } });
    
    // 点击重置按钮
    const resetButton = screen.getByText(/重置/i);
    fireEvent.click(resetButton);
    
    // 验证表单已重置
    expect(screen.getByLabelText(/碳含量/i).value).toBe('');
    expect(screen.getByLabelText(/硅含量/i).value).toBe('');
  });

  test('handles batch prediction mode', async () => {
    render(<PredictionForm />);
    
    // 切换到批量预测标签页
    const batchTab = screen.getByText(/批量预测/i);
    fireEvent.click(batchTab);
    
    // 等待切换完成
    await waitFor(() => {
      expect(screen.getByTestId('mock-upload-button')).toBeInTheDocument();
    });
    
    // 点击上传按钮
    fireEvent.click(screen.getByTestId('mock-upload-button'));
    
    // 验证数据已加载
    expect(screen.getByText(/共2条数据/i) || screen.getByText(/批量预测/i)).toBeInTheDocument();
    
    // 点击开始批量预测
    const batchPredictButton = screen.getByText(/开始批量预测/i);
    fireEvent.click(batchPredictButton);
    
    // 等待预测完成
    await waitFor(() => {
      expect(screen.getByText(/预测结果/i)).toBeInTheDocument();
    });
  });
}); 