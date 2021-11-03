# 0 About
Deepfillv1 reproduction based on PaddlePaddle framework.

This project reproduces [Generative Image Inpainting with Contextual Attention](https://paperswithcode.com/paper/generative-image-inpainting-with-contextual) based on the paddlepaddle framework and participates in the Baidu paper reproduction competition. The AIStudio link is provided as follow:

[link](https://aistudio.baidu.com/aistudio/projectdetail/2547325?channelType=0&channel=0)
# 1 论文简介
- Generative Image Inpainting with Contextual Attention是UIUC的Jiahui Yu在Thomas S. Huang的指导下，联合Adobe Research完成的一项工作，发表于CVPR 2018。

- 作者在Iizuka等人提出的Globally and locally consistent image completion工作的基础上进行改进（**Improved Generative Inpainting Network**），并提出**Contextual Attention**，以利用传统方法中要求图像之中的patch之间存在相似性的思路，弥补卷积神经网络不能有效的从图像较远的区域提取信息的不足。

## 1.1 网络架构
![在这里插入图片描述](https://img-blog.csdnimg.cn/783af11e866e4b12847071cf72a7014f.png?x-oss-process=image,size_20,color_FFFFFF,t_70,g_se,x_16)
- **生成器**：包括两个阶段。第一个阶段是一个**粗糙网络**（Coarse Network），利用**空间衰减重构损失**训练。第二个阶段是一个**细化网络**（Refinement Network），利用**重构损失**和**WGAN损失**训练。
- **判别器**：包括两个部分。第一个部分负责**局部**判别（Local Critic），第二个部分负责**全局**判别（Global Critic），都是基于 **WGAN-GP损失**（带梯度惩罚的WGAN损失）。

## 1.2 **上下文注意力机制**
![在这里插入图片描述](https://img-blog.csdnimg.cn/30a39cb95d2f4aaab4ecead17ad64ef8.png?x-oss-process=image,size_20,color_FFFFFF,t_70,g_se,x_16)
- 思路为从已知图像中借鉴特征信息，以此生成缺失的patch。首先在背景区域提取3x3的patch，并作为卷积核。为了匹配前景（待修复区域）的patch，使用标准化内积（即余弦相似度）来测量，然后用softmax来为每个背景中的patch计算权值，最后选取出一个最好的patch，并反卷积出前景区域。对于反卷积过程中的重叠区域取平均值。
- 通俗一点讲，假设有待修补区域x，通过卷积的方法，从整个图中匹配几个像x的区域a，b，c，d，然后从上述区域中利用softmax找出最像x的区域，最终通过反卷积的方式，来生成x区域的图像。

## 1.3 损失函数
- **WGAN损失**：
![在这里插入图片描述](https://img-blog.csdnimg.cn/3e06672629a44042b431f9d673b94bbe.png)
其中$P_r$是真实的分布，$P_g$是生成数据的分布，这损失在GAN损失的基础上去掉了log。
- **梯度惩罚项**：
![在这里插入图片描述](https://img-blog.csdnimg.cn/b3a3e0df6e034735815591a737fdc5b7.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/1d986227bbfd43a3912d6fbc70ee96ad.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/8a1b2948b8cc495590f1e92a6b9ea7c8.png)
只对位于空洞区域的像素点进行梯度惩罚，利用一个mask实现：![在这里插入图片描述](https://img-blog.csdnimg.cn/1b9342660f7e48989949c50f1d4fb269.png)
- **重构损失**：
![在这里插入图片描述](https://img-blog.csdnimg.cn/ac0f2a63d8ca4114a429448b28d5e8d5.png)
- **空间衰减重构损失**：改变重构损失的mask权重，每一点的权值为$\gamma^{l}$，$\gamma = 0.99$，$l$ 表示该点到已知的像素点最近的距离。

## 1.4 实验
优化器为Adam，学习率为0.0001，batch-size为48，单卡1080Ti训练，在Place2数据集上进心训练，输入图片size为256*256，patch大小为128*128。

- 定性对比
![在这里插入图片描述](https://img-blog.csdnimg.cn/afd3419726494ab889d1133680430f45.png?x-oss-process=image,size_20,color_FFFFFF,t_70,g_se,x_16)
从左往右为原图，输入图片，baseline输出，model输出。
- 定量对比
![在这里插入图片描述](https://img-blog.csdnimg.cn/d137a08b93824db987dcdb7248076a7b.png?x-oss-process=image,size_20,color_FFFFFF,t_70,g_se,x_16)
# 2 GetStarted
## Train

python train.py

the models and optimizer will be saved in checkpoints
you can see the past training log in train.txt

## Test

python test.py

you can change the total test number in test.py
and change the test model in tester.py
you can see the test log in test.txt

## Model
you may find our pre-trained model here:  
link：https://pan.baidu.com/s/16uKEXhe71AxLeOnakQ32Rg 
key：x4tr  

# 3 关于论文
References
1. [论文链接](https://paperswithcode.com/paper/generative-image-inpainting-with-contextual)
2. [论文中文翻译](http://www.gwylab.com/pdf/image-inpainting_chs.pdf)
3. [论文详解](https://www.cnblogs.com/bingmang/p/10000992.html)
4. [论文Github地址](https://github.com/DAA233/generative-inpainting-pytorch)
5. [论文复现Github Repo](https://github.com/JennyVanessa/Paddle-GI)
