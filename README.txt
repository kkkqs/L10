# -*- coding: utf-8 -*-

# 最小平方误差与梯度下降法实验

## 文件说明

### 核心算法
- `task1_algorithms.py`: 广义逆和梯度下降法的实现

### 实验脚本
- `task2_experiment.py`: 任务2 - 数据集m1=[-5,0], m2=[0,5]的分类实验
- `task3_experiment.py`: 任务3 - 数据集m1=[1,0], m2=[0,1]的分类实验  
- `task4_hyperparameters.py`: 任务4 - 超参数调优实验
- `task5_optimizers.py`: 任务5 - 6种优化算法对比

### 运行所有实验
- `run_all.py`: 一键运行所有实验

## 使用方法

单独运行某个任务:
```
python task2_experiment.py
python task3_experiment.py
python task4_hyperparameters.py
python task5_optimizers.py
```

运行所有任务:
```
python run_all.py
```

## 输出结果

各实验会生成相应的图片:
- task2_results.png
- task3_results.png
- task4_hyperparameters.png
- task5_10iterations.png
- task5_50iterations.png
