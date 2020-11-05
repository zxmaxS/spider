pytorch是一个深度学习的依赖，其中实现了很多关于神经网络的内容，从而可以通过少量的代码实现神经网路

# 安装

pytorch通过pip基本没有办法安装，即使下载下来也是各种报错，这里直接推荐使用anaconda，关于anaconda我还没有了解太多，可能之后会写到，首先conda是一个类似pip的包管理工具，但是不止支持Python，还支持其他语言，anaconda就是conda的可视化，首先在开始里面启动anaconda navigator，然后在左面选择environments，这个类似pycharm的虚拟环境，从而方便使用各种版本的python点击下面的create，有的时候网可能会卡，获得不到python的其他版本，这时候在包列表上面有个update，多刷新刷新，然后创建之后选择新创建的环境，点击右面的开始符号，打开终端

anaconda可能是没有换源的原因，无法获取到pytorch的最新版本，直接使用命令行

## 换源

直接复制没啥好说的

~~~
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/bioconda/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/menpo/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
~~~

## 安装

直接安装，这里是两个包，torchvision是一个包含几个数据集以及高级网络模型的扩展包，主要为pytorch服务

~~~
conda install pytorch torchvision
~~~

pytorch安装后引入的是torch

