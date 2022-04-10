COCO API - http://cocodataset.org/

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


COCO is a large image dataset designed for object detection, segmentation, person keypoints detection, stuff segmentation, and caption generation. This package provides Matlab, Python, and Lua APIs that assists in loading, parsing, and visualizing the annotations in COCO. Please visit http://cocodataset.org/ for more information on COCO, including for the data, paper, and tutorials. The exact format of the annotations is also described on the COCO website. The Matlab and Python APIs are complete, the Lua API provides only basic functionality.

In addition to this API, please download both the COCO images and annotations in order to run the demos and use the API. Both are available on the project website.
-Please download, unzip, and place the images in: coco/images/
-Please download and place the annotations in: coco/annotations/
For substantially more details on the API please see http://cocodataset.org/#download.

After downloading the images and annotations, run the Matlab, Python, or Lua demos for example usage.

To install:
-For Matlab, add coco/MatlabApi to the Matlab path (OSX/Linux binaries provided)
-For Python, run "make" under coco/PythonAPI
-For Lua, run “luarocks make LuaAPI/rocks/coco-scm-1.rockspec” under coco/
