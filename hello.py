import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# 生成模拟真实数据：房价与面积的关系
np.random.seed(42)
X = np.random.rand(100, 1) * 100  # 面积 (0-100)
y = 2 * X + np.random.randn(100, 1) * 10  # 房价，带噪声

# 划分训练和测试数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建线性回归模型
model = LinearRegression()
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 绘制图像
plt.scatter(X_train, y_train, color='blue', label='训练数据')
plt.scatter(X_test, y_test, color='green', label='测试数据')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='回归线')
plt.xlabel('面积 (平方米)')
plt.ylabel('房价 (万元)')
plt.title('线性回归：房价预测')
plt.legend()
plt.show()

# 输出模型参数
print(f"斜率 (系数): {model.coef_[0][0]:.2f}")
print(f"截距: {model.intercept_[0]:.2f}")