describe('Login Page', () => {
  beforeEach(() => {
    // 访问登录页面
    cy.visit('http://localhost:9000/login');
  });

  it('displays login form correctly', () => {
    // 验证登录页面元素
    cy.contains('钢铁工业分析系统').should('be.visible');
    cy.get('input[placeholder="用户名"]').should('be.visible');
    cy.get('input[placeholder="密码"]').should('be.visible');
    cy.contains('button', '登录').should('be.visible');
  });

  it('shows error on login with empty credentials', () => {
    // 点击登录按钮但不输入任何信息
    cy.contains('button', '登录').click();
    
    // 验证错误提示
    cy.contains('请输入用户名').should('be.visible');
  });

  it('shows error on login with invalid credentials', () => {
    // 输入错误的用户名密码
    cy.get('input[placeholder="用户名"]').type('wronguser');
    cy.get('input[placeholder="密码"]').type('wrongpass');
    cy.contains('button', '登录').click();
    
    // 验证错误提示（注意：这里基于您的实际应用中的错误消息文本）
    cy.contains('用户名或密码错误').should('be.visible');
  });
  
  it('successfully logs in with valid credentials', () => {
    // 输入有效的用户名密码（使用对应环境的测试账号）
    cy.get('input[placeholder="用户名"]').type('admin');
    cy.get('input[placeholder="密码"]').type('admin');
    cy.contains('button', '登录').click();
    
    // 验证成功后重定向到首页
    cy.url().should('include', '/dashboard');
    // 验证欢迎信息
    cy.contains('欢迎回来').should('be.visible');
  });
}); 