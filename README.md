# inspur

## 文件目录
data/
词向量及一些中间文件。

submit/
测试数据 及 提交文件

model/
模型函数


util/    
config.py  一些文件路径及参数    
cutword.py  :分词    
data2id.py  把词映射成id    
help.py 一些辅助函数，如划分训练验证。     
w2v.py  训练词向量    




## 运行流程：

python  util/w2v.py  
训练词向量。（自动调用分词，data2id）

python train.py cv  cnn1  
对cnn1模型采用交叉验证。

python submit.py  提交结果  
模型选择及cv 记得修改main()函数

