## COCO API - http://cocodataset.org/
# 当前分支在官方 cocoapi 基础上做了优化

说明：官方的cocoapi计算的mAP是IOU从0.5到0.95，每隔0.05下计算的所有类别的AP的平均值；
具体计算过程时是：
1 先将模型的预测框先按照每一张图片，每一种类别，按照置信度从大到小得到maxDet个框
2 然后将测试集中特定类别的所有框按置信度score总的排序，继而再对排序后的指定类别的所有框计算tp、fp，
3 然后对所有类别求平均得到mAP(pr曲线的面积)；
cocoapi原始程序的AP其实是mAP，而且只能计算所有类的mAP，没有计算指定类别的mAP功能

因此也可以知道pr曲线是成反比例，而且置信度score是递减的，置信度score越低，precision越低，recall越高。

注意：maxDet是指一张图片，一种类别容许的最多模型检测框个数，而不是一张图片，所有类别；
如果一张图片中单类别检测框不足maxDet，也直接取单类别所有检测框，并不会其他补0等操作；
cocoapi中矩阵默认-1初始化，因此如果模型检测结果不理想或者测试集中没有满足条件的数据
(比如数据集中只有large的物体，计算small物体的评价指标)都可能出现-1的计算结果。


增加的功能：
1、某一个类别平均精确率的计算；
2、可以指定特定CatID进行指标计算；
3、保存指定条件下的检测好的样本和错检、漏检样本的名称;

具体说明：
除了cocoapi本身的AP(cocoapi原始程序的AP其实是mAP，而且只能计算所有类的mAP，没有计算指定类别的mAP功能)和AR计算之外，

新增功能1：AP是 针对单个类别的
对官方cocoapi修改后新增真正的AP(Average precision)计算值，即修改后的cocoapi输出三种指标，AP平均精确率、
AR平均召回率，mAP：pr曲线围成的面积；

新增功能2：
输出指定类别CatID、指定aRng(small、medium、large)、指定maxDets(每张图每个类别的最多检测框个数)、指定IOUthr下的三个指标值；

新增功能3：这个功能还没完善，有机会完善
除此之外，为了根据模型预测结果分析得到针对性的模型优化方向，可以根据指定条件（CatID、aRng、maxDets、IOUthr）
计算测试集中哪些样本按照指定条件完全检出，并将这些样本名称保存在good_predict.txt文件中，
同时将存在漏检或错检的bbox的样本名称和检测结果分别保存在leak_det_predict.txt和leak_det_predict.json中，
这样就便于进一步分析模型在哪些测试集样本上表现不佳以及表现不佳的原因，进而可以使用离线数据增强或者其他技术对模型进行针对性优化(新增功能3)！！！

参考链接：
1. https://blog.csdn.net/yshtjdx/article/details/111238546?spm=1001.2014.3001.5502
2. https://blog.csdn.net/qq_36302589/article/details/105690491
3. https://zhuanlan.zhihu.com/p/135355550



