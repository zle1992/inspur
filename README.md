# [浪潮杯] 美丽中国-2018全域旅游年  华北赛区一等奖  
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




## 运行流程

python  util/w2v.py  
训练词向量。（自动调用分词，data2id）

python main.py cv  cnn1 trian  
对cnn1模型采用交叉验证。

python main.py cv  cnn1 submit   
对cnn1模型cv提交结果。

