import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def f(x):
    return x * np.cos(0.25 * np.pi * x)

def df(x):
    return np.cos(0.25 * np.pi * x) - 0.25 * np.pi * x * np.sin(0.25 * np.pi * x)

def gradient_descent(x0, lr, iterations):
    x_history = [x0]
    for _ in range(iterations):
        x = x_history[-1]
        x_new = x - lr * df(x)
        x_history.append(x_new)
    return np.array(x_history)

def sgd(x0, lr, iterations):
    x_history = [x0]
    for _ in range(iterations):
        x = x_history[-1]
        noise = np.random.normal(0, 1)
        x_new = x - lr * (df(x) + noise)
        x_history.append(x_new)
    return np.array(x_history)

def adagrad(x0, lr, iterations, eps=1e-6):
    x_history = [x0]
    G = 0
    for _ in range(iterations):
        x = x_history[-1]
        grad = df(x)
        G += grad ** 2
        x_new = x - lr / (np.sqrt(G) + eps) * grad
        x_history.append(x_new)
    return np.array(x_history)

def rmsprop(x0, lr, iterations, alpha=0.9, eps=1e-6):
    x_history = [x0]
    E = 0
    for _ in range(iterations):
        x = x_history[-1]
        grad = df(x)
        E = alpha * E + (1 - alpha) * grad ** 2
        x_new = x - lr / (np.sqrt(E) + eps) * grad
        x_history.append(x_new)
    return np.array(x_history)

def momentum(x0, lr, iterations, lam=0.9):
    x_history = [x0]
    v = 0
    for _ in range(iterations):
        x = x_history[-1]
        grad = df(x)
        v = lam * v + lr * grad
        x_new = x - v
        x_history.append(x_new)
    return np.array(x_history)

def adam(x0, lr, iterations, beta1=0.9, beta2=0.999, eps=1e-6):
    x_history = [x0]
    m = 0
    v = 0
    for t in range(1, iterations + 1):
        x = x_history[-1]
        grad = df(x)
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad ** 2
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        x_new = x - lr * m_hat / (np.sqrt(v_hat) + eps)
        x_history.append(x_new)
    return np.array(x_history)

x0 = -4
lr = 0.4
iterations_10 = 10
iterations_50 = 50

np.random.seed(42)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

methods_10 = [
    (gradient_descent(x0, lr, iterations_10), 'GD'),
    (sgd(x0, lr, iterations_10), 'SGD'),
    (adagrad(x0, lr, iterations_10), 'Adagrad'),
    (rmsprop(x0, lr, iterations_10), 'RMSProp'),
    (momentum(x0, lr, iterations_10), 'Momentum'),
    (adam(x0, lr, iterations_10), 'Adam')
]

for idx, (x_hist, name) in enumerate(methods_10):
    ax = axes[idx // 3, idx % 3]
    x_range = np.linspace(-5, 5, 200)
    ax.plot(x_range, f(x_range), 'gray', alpha=0.5)
    ax.plot(x_hist, f(x_hist), 'ro-', markersize=4)
    for i, (xi, yi) in enumerate(zip(x_hist, f(x_hist))):
        ax.text(xi, yi, str(i), fontsize=8)
    ax.set_title(f'{name} (10次)')
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.grid(True)

plt.tight_layout()
plt.savefig('task5_10iterations.png', dpi=150)
plt.show()

np.random.seed(42)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

methods_50 = [
    (gradient_descent(x0, lr, iterations_50), 'GD'),
    (sgd(x0, lr, iterations_50), 'SGD'),
    (adagrad(x0, lr, iterations_50), 'Adagrad'),
    (rmsprop(x0, lr, iterations_50), 'RMSProp'),
    (momentum(x0, lr, iterations_50), 'Momentum'),
    (adam(x0, lr, iterations_50, beta1=0.99), 'Adam (β1=0.99)')
]

for idx, (x_hist, name) in enumerate(methods_50):
    ax = axes[idx // 3, idx % 3]
    x_range = np.linspace(-5, 5, 200)
    ax.plot(x_range, f(x_range), 'gray', alpha=0.5)
    ax.plot(x_hist, f(x_hist), 'bo-', markersize=3, alpha=0.6)
    ax.plot(x_hist[0], f(x_hist[0]), 'go', markersize=8, label='起点')
    ax.plot(x_hist[-1], f(x_hist[-1]), 'ro', markersize=8, label='终点')
    ax.set_title(f'{name} (50次)')
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.savefig('task5_50iterations.png', dpi=150)
plt.show()

print("10次迭代最终结果:")
np.random.seed(42)
for name, func in [('GD', gradient_descent), ('SGD', sgd), ('Adagrad', adagrad), 
                   ('RMSProp', rmsprop), ('Momentum', momentum), ('Adam', adam)]:
    if name == 'RMSProp':
        x_hist = func(x0, lr, iterations_10)
    elif name == 'Momentum':
        x_hist = func(x0, lr, iterations_10)
    else:
        x_hist = func(x0, lr, iterations_10)
    print(f"{name:10s}: x={x_hist[-1]:7.4f}, f(x)={f(x_hist[-1]):7.4f}")
