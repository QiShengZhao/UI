module.exports = {
  testEnvironment: 'jsdom',
  moduleNameMapper: {
    "\\.(css|less|scss|sass)$": "identity-obj-proxy",
    "\\.(jpg|jpeg|png|gif|webp|svg)$": "<rootDir>/test/__mocks__/fileMock.js"
  },
  setupFilesAfterEnv: ['<rootDir>/test/setupTests.js'],
  testPathIgnorePatterns: ['/node_modules/', '/dist/'],
  testMatch: ['<rootDir>/test/**/*.test.js', '<rootDir>/test/**/*.test.jsx'],
  transform: {
    "^.+\\.(js|jsx)$": "babel-jest"
  },
  transformIgnorePatterns: [
    "/node_modules/(?!(@testing-library)/)"
  ]
}; 