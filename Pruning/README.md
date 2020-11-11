## Pruning

### Unstructured Top-k Pruning
- 基本说明： 
  - mnist-MLP 模型 
  - Global Top-k Pruning;
  - 进阶参考 https://github.com/Eric-mingjie/rethinking-network-pruning

- 步骤：
  1. 训练Baseline 浮点模型， 运行 `python mnist_test.py`, 令其中 `Baseline = True`, 输出精度 保存模型checkpoint;
  2. 稀疏化模型，令 `mnist_test.py` 中 `Prune = True`， 并得到测试精度；
  3. 调整  `mnist_test.py` 中的`prune ratio`;
  4. 若精度不满足要求，则进行重训练; 设置 `Retrain=True`.
