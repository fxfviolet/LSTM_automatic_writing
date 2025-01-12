# LSTM自动成文
LSTM，长短时记忆网络（Long short term memory）是一种循环神经网络（Recurrent neural network）。由于独特的设计结构，LSTM适合于处理和预测时间序列中间隔和延迟非常长的重要事件。LSTM具有非常广泛的应用，包括语音识别、文本分类、语言模型、自动对话、机器翻译、图像标注等领域。这里介绍用LSTM自动生成文本。


## 1 &nbsp;算法
#### 算法原理：
循环神经网络会对每一个时刻的输入结合当前模型的状态给出一个输出。RNN的单元为对当前输入、过去时刻输入分配权重，即各个输入在网络层中所占的比重，通过使用损失函数动态地调整分配权重。LSTM是一种拥有三个“门”结构的特殊网络结构，依靠“门”结构让信息有选择性地影响循环神经网络中每个时刻的状态。

#### 训练方法：
对训练文本分词，创建词汇表，统计词语频数，以频数从高到低排序的编号作为词语的编号。遍历文本词语列表，对每一个词语转为编号。根据词语编号，创建矩阵，生成训练用的batch和label数据。接下来设置LSTM单元，首先定义单个基本的LSTM单元，再接入一个Dropout层，实现深层循环神经网络中每一个时刻的前向传播过程，构造完多层LSTM后，对状态进行初始化，再创建递归神经网络。将输出张量串接到一起并转为一维向量。定义权重和偏置，乘上权重加上偏置得到网络的输出。最后用交叉熵计算损失，进行优化。

训练数据可以任何形式（小说、诗歌、新闻等）的文本。

训练代码：[train.py](https://github.com/fxfviolet/LSTM_automatic_writing/blob/master/train.py) 

```
python train.py \
  --use_embedding True \
  --input_file data/novel.txt \
  --num_steps 80 \
  --name novel \
  --learning_rate 0.005 \
  --num_seqs 32 \
  --num_layers 3 \
  --embedding_size 256 \
  --lstm_size 256 \
  --max_steps 1000000
```

生成文本代码：[sample.py](https://github.com/fxfviolet/LSTM_automatic_writing/blob/master/sample.py) 

```
python sample.py \
  --converter_path model/converter.pkl \
  --checkpoint_path  model \
  --max_length 500  \
  --use_embedding \
  --num_layers 3 \
  --start_string 
```

 
## 2 &nbsp;生成文本
##### 以下段落是用算法自动生成的，可以按需求设置段落长度和起始词语。
前两天才停了两天的雨，一早起来又细细密密地拉开帷幕。这样的雨天，路上行人稀少，整条街道冷冷清清的。只有几个挑着担子的菜贩穿着雨衣在街头晃悠。昨天下午，老公接到一个电话，有位朋友想请他帮忙安装一个水塔。老公望着外面连绵不断的雨雾问他，是装在室外还是室内，如果是室外，这样的天气恐怕不好。晚上又在下雨，有一位朋友来家里坐。两个男人坐在一块儿天南地北闲聊着，不知怎的，话题扯到一个与我们做同样生意的一个同行熟人身上，是朋友的远房叔叔...

遇到所有的事情都能做到不动声色，于是我们学会了掩饰和伪装，即使想哭也会笑着说出一些难过，用一种平淡或者遥远的语气，诉说着一件对别人不重要却对自己很刻骨铭心的事情。在感情里，遗憾开始总是甜蜜的后来就有了厌倦、习惯、背弃、寂寞、绝望和冷笑曾经渴望与一个，无论是身还是心。泪眼朦胧，无语凝噎。“你见感情放过谁”,我们不知道他现在是否已经走了出来，可是那个女孩一定就如同一颗烙印一样深深的印在他心里，感情这个东西，谁也救不了你，你只能自救。无论你喜欢的那个人不喜欢你，还是你想挽回一段感情，别人再怎么劝都是没用的。是啊，这个世界上真正的爱情，本来就应该是门当户对的，不是物质上的门当户对，而是精神上的聊得来。优秀的人不会让人感到触不可及，不会不尊重别人。会努力让对方活得更像她自己。
  
