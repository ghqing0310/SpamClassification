# SpamClassification
基于CNN的垃圾邮件分类系统

### 数据处理与准备

与参考数据集不同，我选择的数据集是中文邮件。源数据见data/mailContent_list.pickle和mailLabel_list.pickle，共有64620封邮件，其中垃圾邮件有42854封。

考虑到GPU显存不足，故只随机选取其中1000封，保存为data/mailContent_list_1000.pickle和mailLabel_list_1000.pickle，其中垃圾邮件有659封。

随机查看一封垃圾邮件：

![img](file:///C:/Users/guoha/AppData/Local/Temp/msohtmlclip1/01/clip_image002.png)

随机查看一封非垃圾邮件：

![img](file:///C:/Users/guoha/AppData/Local/Temp/msohtmlclip1/01/clip_image004.png)

统计得单封邮件最大字数为23945，平均字数为740，中位数为336；邮件中出现的汉字共为4427个。

处理数据集的代码文件见python/data.py中的TextData类：

（1）  \__init__初始化：划分训练集和测试集（7:3），定义类数、词向量维度和单封邮件最大长度（根据经验，1000个字足够判断是否为垃圾邮件）。

（2）  __prepare_data获取基本信息：获取词汇表（根据出现次数倒序排列）并获得词汇量；构建词-id的映射。

（3）  __content2X构建输入数据：邮件长度不足1000则在前添加padding；超过则在前删去。

（4）  __label2Y构建输出数据：将label转为0和1。

（5）  get_data获取数据：外部调用，返回训练输入、训练输出、测试输入、测试输出。

<img src="file:///C:/Users/guoha/AppData/Local/Temp/msohtmlclip1/01/clip_image006.png" alt="img" style="zoom: 50%;" />

调用见python/main.py：

<img src="file:///C:/Users/guoha/AppData/Local/Temp/msohtmlclip1/01/clip_image008.png" alt="img" style="zoom:50%;" />



### 模型构建

模型构建见python/cnn.py：

（1）  初始化随机词向量：将每封邮件的词id列表转化为词嵌入矩阵，(train_size, seq_length)->(train_size, seq_length, embedding_dim)。

（2）  CNN卷积层：Conv1d，设定输入通道数、输出通道数和核大小，(train_size, embedding_dim, seq_length)->(train_size, num_filters, conv(seq_length))。

（3）  Max Pooling最大池化层：(train_size, num_filters, conv(seq_length))->(train_size, num_filters, 1)，取切片转化为(train_size, num_filters)。

（4）  全连接层：设定隐藏层维度，(train_size, num_filters)->(train_size, hidden_dim)。

（5）  Dropout层：设定保留概率。

（6）  relu激活函数层。

（7）  全连接层：(train_size, hidden_dim)->(train_size, num_classes)。

<img src="file:///C:/Users/guoha/AppData/Local/Temp/msohtmlclip1/01/clip_image010.png" alt="img" style="zoom:50%;" />



### 训练过程

模型构建见python/train.py：

（1）  \__init__初始化：设置随机种子，初始化模型。

（2）  __cnn_build构建模型：将模型放入GPU，定义交叉熵损失函数和SGD优化器。

（3）  cnn_train训练模型：外部调用，设定epoch和batch_size，保存最优模型。

（4）  cnn_test测试模型：外部调用，获得测试集结果。

调用见python/main.py：<img src="file:///C:/Users/guoha/AppData/Local/Temp/msohtmlclip1/01/clip_image012.png" alt="img" style="zoom:50%;" />

考虑GPU显存容量，设定batch_size为100，epoch为400。

<img src="file:///C:/Users/guoha/AppData/Local/Temp/msohtmlclip1/01/clip_image014.png" alt="img" style="zoom:50%;" />

训练结果为：

<img src="file:///C:/Users/guoha/AppData/Local/Temp/msohtmlclip1/01/clip_image016.png" alt="img" style="zoom:50%;" />



### 测试集结果

测试集结果见python/train.py：

（1）  evaluate：计算loss和accuracy。

（2）  print_report_table：计算Precision、Recall、F1和Support。

（3）  print_confusion_matrix：计算混淆矩阵。

（4）  plot_roc_curve：绘制ROC曲线。

<img src="file:///C:/Users/guoha/AppData/Local/Temp/msohtmlclip1/01/clip_image018.png" alt="img" style="zoom:50%;" />

调用见python/main.py：

**<img src="file:///C:/Users/guoha/AppData/Local/Temp/msohtmlclip1/01/clip_image020.png" alt="img" style="zoom:50%;" />**

测试结果为：

<img src="file:///C:/Users/guoha/AppData/Local/Temp/msohtmlclip1/01/clip_image022.png" alt="img" style="zoom:50%;" />

<img src="file:///C:/Users/guoha/AppData/Local/Temp/msohtmlclip1/01/clip_image024.png" alt="img" style="zoom:50%;" />
