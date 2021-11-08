# 数字图像

## 1. 图像

**像素：** *像素是分辨率的单位。像素是构成位图图像的基本单元，每个像素都有自己的颜色。*

**分辨率：** *又称“解析度”，图像的分辨率就是单位英寸内的像素点数。单位是PPI（Pixels Per Inch）*

> PPI表示的是每英寸对角线上所拥有的像素数目（w：宽度像素数，h：高度像素数，di：屏幕尺寸），屏幕尺寸指的是对角线长度

![image-20211108173426294](img\image-20211108173426294.png)

> 在生活中经常混淆分辨率与清晰度的关系以及分辨率与屏幕尺寸的关系。

**灰度：** *表示图像像素明暗程度的数值，也就是黑白图像中点的颜色深度。范围一般为0-255.白色为255，黑色为0.*

**通道：** *把图像分解成一个或多个颜色成分：*

- 单通道：一个像素点只需一个数值表示，只能表示灰度，0为黑色；（二值图&灰度图）
- 三通道：RGB模式，把图像分为红绿蓝三个通道，可以表示彩色，全0表示黑色；
- 四通道：RGBA模式，把RGB基础上加上alpha通道，表示透明度，alpha=0表示全透明；

**对比度：** *指不同颜色之间的差别。对比度=最大灰度值/最小灰度值*

**RGB转换为Gray（降维、保留梯度信息）：** *常见以下几种算法* 

- 浮点算法：`Gray = 0.3R + 0.59G + 0.11B`
- 整数方法：`Gray = ( 30R + 59G + 11B ) / 100`
- 移位方法：`Gray = ( 76R + 151G + 28B ) >> 8 `
- 平均值法：`Gray = ( R + G + B ) / 3`
- 保留绿色：` Gray = G`

**RGB值化为浮点数：** 

- 浮点数运算结果更精确，整数运算中会因丢弃小数部分可能导致颜色值严重失真，计算过程越多越失真
- 将RGB值转化为[0,1]浮点数 （除以255）
- 二值化：先转浮点数

```python
if(img_gray[i,j]<=0.5):
	img_gray[i,j] = 0
else
	img_gray[i,j] = 1
```

- opencv大坑之BGR：opencv对于读进来的图片的通道排列是BGR，而不是主流的RGB！谨记！

```python
#opencv读入的矩阵是BGR，如果想转为RGB，可以这么转
img4 = cv2.imread('1.jpg')
img4 = cv2.cvtColor(img4,cv2.COLOR_BGR2RGB)
```

**频率：** *灰度值变化剧烈程度的指标，是灰度在平面空间上的梯度。*

**幅值：** *幅值是在一个周期内，交流电瞬时出现的最大绝对值，也就是一个正弦波，波峰到波谷的距离的一半。*

## 2. 图像的取样与量化

**数字图像：** *计算机保存的图像都是一个一个的像素点，称为数字图像*

**取样（数字化坐标值）：** *就是要用多少点来描述一幅图像，取样结果质量的高低就是用分辨率来衡量的*

**量化（数字化幅度值）：** *是指要使用多大范围的数值来表示图像采样之后的一个点。*

## 3.上采样与下采样

**下采样：** *缩小图像（或称为下采样(subsampled)或者降采样(downsampled)）*

*降采样的主要目的有两种：*

- 使得图像符合显示区域的大小
- 生成对应的略缩图

> 下采样原理：`( M / s ) * ( N / s )`

**上采样：** *放大图像*

> 上采样原理：内插值

**常用的插值方法**

- 最邻近插值（The nearest interpolation）
- 双线性插值

> 横竖两次单线性插值
>
> 图像相邻四个点，故而分母都是1
>
> 存在的问题：
>
> 坐标系的选择  解决办法（找几何中心：+0.5）
>
> 要通过双线性插值的方法算出
>
> 双线性插值的计算灰度是连续的，更光滑

*由单线性插值引入：*

![image-20211108172939177](img\image-20211108172939177.png)

*引出双线性插值：实质上是x方向做了两次单线性插值，y方向上做了一次单线性插值。*



![math_double](C:\Users\moresweet\Desktop\cv_learn\2.digitalImage\img\math_double.png)

> 按照原理可以得出
>
> `srcX = ( dstX ) * ( srcWidth / dstWidth )`
>
> `srcY = ( dstY ) * ( srcHeight / dstHeight ) `
>
>  这样会有问题，如果源图像和目的图像的映射均选择映射的左上角的原点，那么会导致插值的结果取值偏左上，若都选择右下角，则又会偏右下。
>
> 为了保证取值的效果，需要取图像中心，经过数学计算证明，原坐标和目的坐标均+0.5，即可保证取值在图像中心，插值效果被优化。双线性插值与最邻近插值比计算量要大很多，但不会存在灰度不连续的缺点，故而图像看起来更光滑。

## 4. 直方图

> 描述对应灰度值的像素数

![image-20211108190356853](img\image-20211108190356853.png)

**特点：** 

- 直方图不关心像素所处的空间位置，因此不受图像旋转和平移变化的影响，可以作为图像的特征
- 任何一幅特定的图像都有唯一的直方图与之对应，但不同的图像可以有相同的直方图与之对应
- 两个不连续区域的要相加

**应用：** *推测图像的亮度分布程度*

![image-20211108190529379](img\image-20211108190529379.png)

**直方图均衡化：** *用一定的算法使直方图大致平和的方法*

> 直方图均衡化的作用是图像增强。

![image-20211108190826547](img\image-20211108190826547.png)

*方法：*

- 依次扫描原始灰度图像的每一个像素，计算出图像的灰度直方图H
- 计算灰度直方图的累加直方图
- 根据累加直方图和直方图均衡化的原理得到输入与输出的映射关系
- 最后根据映射关系得到结果：`dst( x, y ) = H'( src( x, y ) )`

*公式解释：*

![image-20211108191606151](img\image-20211108191606151.png)

## 5. 图像处理方法

### 1. 直方图均衡化 

> 同上

### 2. 滤波

*滤波是图像处理的基本方法*

![image-20211108193427191](img\image-20211108193427191.png)

### 3. 卷积

 *卷积的原理和滤波的原理几乎一样，但是卷积操作在做乘积之前，需要先将卷积核翻转180度，之后再做乘积。*

*卷积的数学定义（g为作用在f上的filter或者kernel）：*

![image-20211108193702152](img\image-20211108193702152.png)

> 卷积负责提取图像中的局部特征

**过滤器：**

- 滤波器的大小应该是奇数，这样它才有一个中心，这压根也就有了核的半径。
- 滤波器矩阵的元素之和为1，保证了滤波前后的图像亮度不变。（非必须）
- 元素之和大于1则滤波后的图像比原图更亮；反之则变暗。（注：和为0图像并不会变黑，只是很暗）
- 滤波后的结构，可能会出现负数或者大于255的数值。处理方法为截断到0-255，对于负数可以取绝对值处理。

**卷积核/Kernel：**

- 每个卷积核都代表了一种图像模式

**常见的卷积核：**

- 黑白边界

![image-20211108195046211](img\image-20211108195046211.png)

- 原图卷积（无用）

![image-20211108195114753](img\image-20211108195114753.png)

- 平滑（平滑均值滤波）

![image-20211108195147190](img\image-20211108195147190.png)

- 高斯平滑

![image-20211108195223281](img\image-20211108195223281.png)

- 图像锐化（拉普拉斯变换核函数）

![image-20211108195255758](img\image-20211108195255758.png)

- Soble边缘检测（强调了边缘相邻的像素点对边缘的影响）

![image-20211108195321214](img\image-20211108195321214.png)

**步长：**

*如果用(k,k)的过滤器来卷积一张(h,w)的图片，每次移动一个像素的话，得出的输出结果是(h-k+1,w-k+1)，k是过滤器大小，h和w分别是图片的高宽。*

*若每次移动s个像素，称为**步长**为s的卷积，输出结果为((h-k)/s + 1, (w-k)/s +1)*

- k或者s的值比1大的话，那么每次卷积之后的结果的宽要比卷积前小一些。
- 丢失信息

**填充（Padding）：**

*卷积核在对边缘卷积时，会丢失信息，故应在源图像上填充0，以减少信息丢失*

> p = (s(h - 1) - h + k) / 2
>
> 若步长为1
>
> p = (k - 1) / 2
>
> p: 增加p层0

*填充的三种模式*

- full

![image-20211108214847918](img\image-20211108214847918.png)

- same

> 中心点与边角重合卷积，same，一般same输出的特征图尺寸与原图保持一致，当然这与步长也有关系

![image-20211108214935707](img\image-20211108214935707.png)

- valid

![image-20211108215002002](img\image-20211108215002002.png)

*输入不符合尺寸的解决办法：*

- 插值
- resize
- 加填充padding

**三通道卷积过程**：

![image-20211108215106260](img\image-20211108215106260.png)

**卷积核的确定：**

> 通过CNN训练而得
