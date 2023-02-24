# semi-automatic-watermark-removal
Project for detecting and removing visible watermarks for images

> Unofficial implementation and application of 《On The Effectiveness Of Visible Watermarks》
>
> Based on project [automatic-watermark-detection](https://github.com/rohitrango/automatic-watermark-detection)
>
> Source: 浙江大学SRTP项目——水印检测与去除
>
> Contributors: Xiaodan Xu, Yining Mao, Hanjing Zhou

## 说明

本项目基于GitHub项目[automatic-watermark-detection](https://github.com/rohitrango/automatic-watermark-detection)修改，在此对项目作者表示感谢！

automatic-watermark-detection项目是对paper "On The Effectiveness Of Visible Watermarks"的非官方复现。我们在项目作者工作的基础上，完成了随机位置水印检测与去除、满屏水印检测与去除的工作。

项目包括【满屏水印检测与去除】、【单个随机位置水印检测与去除】两个部分。


## 我们的其他去水印工作

我们的另一个项目使用cGAN去水印：

[watermark-detection-and-removal](https://github.com/doriscullen/watermark-detection-and-removal)

此项目为《Towards Photo-Realistic Visible Watermark Removal with Conditional Generative Adversarial Networks》一文的非官方复现。



## 水印研究数据集

水印研究数据集包含15000张带文字水印图和1000张带满屏水印图

您可以从百度网盘下载我们创建的水印研究数据集：

【链接：https://pan.baidu.com/s/14Ap00RcrfhlEsO3WBB62_Q	提取码：syyj】



## 本项目使用说明

**满屏水印**

源码：multiple_watermark.ipynb

工作目录：multiple

其中train放训练集；test放测试集，test中的patch作为临时文件夹，存储去掉水印的patches；result_test存放测试结果。

**随机位置单个水印**

源码：single_random_watermark.ipynb

工作目录：random

其中train放训练集，在estimate文件夹下仅存放1张图，用于手动框选水印位置；test放测试集，test中的patch作为临时文件夹，存储去掉水印的patches；result_test存放测试结果。

### 1 准备数据集

请参照我们对文件的命名格式，您可使用我们在另一个项目[watermark-detection-and-removal](https://github.com/doriscullen/watermark-detection-and-removal)中的加水印程序，或下载我们的“水印研究数据集”。如果使用不同的文件命名规则，请修改程序中读取文件相关的部分。

### 2 执行程序

使用Jupyter notebook打开对应的.ipynb文件，按步执行即可。



## 展望

- 水印检测方法有待优化
- 去水印效果还不完美，留有一定的水印痕迹
