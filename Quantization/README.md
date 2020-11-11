## Quantization 

### Basic Fixed-Point Quantization
- 基本说明： 
  - mnist-MLP 模型 
  - 定点量化，目前仅支持所有的层采用相同量化配置;
  - 量化代码见: modules/

- 步骤：
  1. 训练Baseline 浮点模型， 运行 `python mnist_test.py`, 令其中 `Quant=False`, 输出精度 保存模型checkpoint;
  2. 量化模型，令 `mnist_test.py` 中 `Quant=True`， 并得到测试精度；
  3. 调整 `modules/qlayer.py QLinear()`中的量化位宽设置, 直至精度合理;
  4. 若精度不满足要求，则进行重训练; 设置 `Retrain=True`.

### Todo
- [ ] 动态量化方案;
- [ ] Quantization-Aware-Training
- [ ] Post-Training Quantization