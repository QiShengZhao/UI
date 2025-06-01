describe('Prediction Feature', () => {
  beforeEach(() => {
    // 模拟登录过程
    cy.visit('http://localhost:9000/login');
    cy.get('input[placeholder="用户名"]').type('admin');
    cy.get('input[placeholder="密码"]').type('admin');
    cy.contains('button', '登录').click();
    
    // 等待登录完成并导航到预测页面
    cy.url().should('include', '/dashboard');
    cy.get('a[href="/prediction"]').click();
    cy.url().should('include', '/prediction');
  });

  it('displays prediction form correctly', () => {
    // 验证表单元素存在
    cy.contains('钢材性能预测').should('be.visible');
    cy.contains('单条预测').should('be.visible');
    
    // 检查输入字段
    cy.get('input[id*="carbon"]').should('be.visible');
    cy.get('input[id*="silicon"]').should('be.visible');
    cy.get('input[id*="manganese"]').should('be.visible');
    cy.get('input[id*="exitTemperature"]').should('be.visible');
    
    // 验证按钮
    cy.contains('button', '开始预测').should('be.visible');
    cy.contains('button', '重置').should('be.visible');
  });

  it('submits prediction form and displays results', () => {
    // 填写表单
    cy.get('input[id*="carbon"]').type('0.15');
    cy.get('input[id*="silicon"]').type('0.25');
    cy.get('input[id*="manganese"]').type('1.4');
    cy.get('input[id*="exitTemperature"]').type('950');
    cy.get('input[id*="thickness"]').type('15');
    cy.get('input[id*="coolingTemperature"]').type('600');
    
    // 提交表单
    cy.contains('button', '开始预测').click();
    
    // 等待预测结果显示
    cy.contains('预测结果').should('be.visible', { timeout: 10000 });
    
    // 验证图表和预测值显示
    cy.get('.ant-card').contains('屈服强度').should('be.visible');
    cy.get('.ant-card').contains('抗拉强度').should('be.visible');
    cy.get('.ant-card').contains('伸长率').should('be.visible');
    cy.get('.ant-card').contains('韧性').should('be.visible');
  });

  it('resets form when reset button is clicked', () => {
    // 填写表单
    cy.get('input[id*="carbon"]').type('0.15');
    cy.get('input[id*="silicon"]').type('0.25');
    
    // 点击重置按钮
    cy.contains('button', '重置').click();
    
    // 验证输入字段已重置
    cy.get('input[id*="carbon"]').should('have.value', '');
    cy.get('input[id*="silicon"]').should('have.value', '');
  });

  it('switches to batch prediction mode', () => {
    // 点击批量预测标签
    cy.contains('批量预测').click();
    
    // 验证批量预测界面显示
    cy.contains('上传CSV文件').should('be.visible');
    cy.contains('button', '开始批量预测').should('be.visible');
  });
}); 