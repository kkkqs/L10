import numpy as np
import matplotlib.pyplot as plt
from task1_algorithms import PseudoInverse, GradientDescent
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

np.random.seed(42)

m1 = np.array([-5, 0])
m2 = np.array([0, 5])
cov = np.eye(2)

X1 = np.random.multivariate_normal(m1, cov, 200)
X2 = np.random.multivariate_normal(m2, cov, 200)
y1 = np.ones(200)
y2 = -np.ones(200)

X = np.vstack([X1, X2])
y = np.hstack([y1, y2])

indices = np.random.permutation(400)
train_idx = indices[:320]
test_idx = indices[320:]

X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]

pi_model = PseudoInverse()
pi_model.fit(X_train, y_train)
pi_train_acc = np.mean(pi_model.predict(X_train) == y_train)
pi_test_acc = np.mean(pi_model.predict(X_test) == y_test)

gd_model = GradientDescent(lr=0.01, epochs=500, batch_size=32)
gd_model.fit(X_train, y_train)
gd_train_acc = np.mean(gd_model.predict(X_train) == y_train)
gd_test_acc = np.mean(gd_model.predict(X_test) == y_test)

print(f"广义逆 - 训练集准确率: {pi_train_acc:.4f}, 测试集准确率: {pi_test_acc:.4f}")
print(f"梯度下降 - 训练集准确率: {gd_train_acc:.4f}, 测试集准确率: {gd_test_acc:.4f}")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

for idx, (model, title) in enumerate([(pi_model, '广义逆'), (gd_model, '梯度下降')]):
    Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    axes[idx].contourf(xx, yy, Z, levels=[-10, 0, 10], alpha=0.3, colors=['blue', 'red'])
    axes[idx].contour(xx, yy, Z, levels=[0], colors='black', linewidths=2)
    axes[idx].scatter(X1[:, 0], X1[:, 1], c='red', marker='o', label='+1类', alpha=0.6)
    axes[idx].scatter(X2[:, 0], X2[:, 1], c='blue', marker='s', label='-1类', alpha=0.6)
    axes[idx].set_title(f'{title}分类面')
    axes[idx].legend()
    axes[idx].set_xlabel('x1')
    axes[idx].set_ylabel('x2')

axes[2].plot(gd_model.losses)
axes[2].set_title('梯度下降损失曲线')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('Loss')
axes[2].grid(True)

plt.tight_layout()
plt.savefig('task2_results.png', dpi=150)
plt.show()
