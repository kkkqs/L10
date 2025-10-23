# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from task1_algorithms import GradientDescent
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

np.random.seed(42)

def generate_data(m1, m2, n_samples=200):
    X1 = np.random.multivariate_normal(m1, np.eye(2), n_samples)
    X2 = np.random.multivariate_normal(m2, np.eye(2), n_samples)
    X = np.vstack([X1, X2])
    y = np.hstack([np.ones(n_samples), -np.ones(n_samples)])
    indices = np.random.permutation(2 * n_samples)
    train_idx = indices[:int(0.8 * 2 * n_samples)]
    test_idx = indices[int(0.8 * 2 * n_samples):]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]

m1 = np.array([-5, 0])
m2 = np.array([0, 5])

configs = [
    {'lr': 0.001, 'epochs': 200, 'batch_size': 32, 'name': 'lr=0.001, bs=32'},
    {'lr': 0.01, 'epochs': 200, 'batch_size': 32, 'name': 'lr=0.01, bs=32'},
    {'lr': 0.01, 'epochs': 200, 'batch_size': 16, 'name': 'lr=0.01, bs=16'},
    {'lr': 0.01, 'epochs': 500, 'batch_size': 32, 'name': 'lr=0.01, ep=500'},
]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for idx, config in enumerate(configs):
    X_train, y_train, X_test, y_test = generate_data(m1, m2)
    
    model = GradientDescent(lr=config['lr'], epochs=config['epochs'], batch_size=config['batch_size'])
    model.fit(X_train, y_train)
    
    train_acc = np.mean(model.predict(X_train) == y_train)
    test_acc = np.mean(model.predict(X_test) == y_test)
    
    axes[idx].plot(model.losses)
    axes[idx].set_title(f"{config['name']}\n训练:{train_acc:.3f}, 测试:{test_acc:.3f}")
    axes[idx].set_xlabel('Epoch')
    axes[idx].set_ylabel('Loss')
    axes[idx].grid(True)

plt.tight_layout()
plt.savefig('task4_hyperparameters.png', dpi=150)
plt.show()

print("\n样本数量对比:")
for n in [100, 200, 500]:
    X_train, y_train, X_test, y_test = generate_data(m1, m2, n)
    model = GradientDescent(lr=0.01, epochs=200, batch_size=32)
    model.fit(X_train, y_train)
    train_acc = np.mean(model.predict(X_train) == y_train)
    test_acc = np.mean(model.predict(X_test) == y_test)
    print(f"样本数={n}: 训练准确率={train_acc:.4f}, 测试准确率={test_acc:.4f}")
