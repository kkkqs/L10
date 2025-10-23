# -*- coding: utf-8 -*-
import subprocess
import sys

tasks = [
    ("任务2: m1=[-5,0], m2=[0,5]", "task2_experiment.py"),
    ("任务3: m1=[1,0], m2=[0,1]", "task3_experiment.py"),
    ("任务4: 超参数调优", "task4_hyperparameters.py"),
    ("任务5: 6种优化算法对比", "task5_optimizers.py"),
]

print("="*60)
print("模式识别实验 - 最小平方误差和梯度下降法")
print("="*60)

for title, script in tasks:
    print(f"\n{'='*60}")
    print(f"运行 {title}")
    print(f"{'='*60}")
    subprocess.run([sys.executable, script])
    print(f"\n{title} 完成!")

print("\n" + "="*60)
print("所有实验完成!")
print("="*60)
