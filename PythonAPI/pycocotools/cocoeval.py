__author__ = 'tsungyi'

import numpy as np
import datetime
import time
from collections import defaultdict
from . import mask as maskUtils
import copy

class COCOeval:
    # Interface for evaluating detection on the Microsoft COCO dataset.
    #
    # The usage for CocoEval is as follows:
    #  cocoGt=..., cocoDt=...       # load dataset and results
    #  E = CocoEval(cocoGt,cocoDt); # initialize CocoEval object
    #  E.params.recThrs = ...;      # set parameters as desired
    #  E.evaluate();                # run per image evaluation
    #  E.accumulate();              # accumulate per image results
    #  E.summarize();               # display summary metrics of results
    # For example usage see evalDemo.m and http://mscoco.org/.
    #
    # The evaluation parameters are as follows (defaults in brackets):
    #  imgIds     - [all] N img ids to use for evaluation
    #  catIds     - [all] K cat ids to use for evaluation
    #  iouThrs    - [.5:.05:.95] T=10 IoU thresholds for evaluation
    #  recThrs    - [0:.01:1] R=101 recall thresholds for evaluation
    #  areaRng    - [...] A=4 object area ranges for evaluation
    #  maxDets    - [1 10 100] M=3 thresholds on max detections per image
    #  iouType    - ['segm'] set iouType to 'segm', 'bbox' or 'keypoints'
    #  iouType replaced the now DEPRECATED useSegm parameter.
    #  useCats    - [1] if true use category labels for evaluation
    # Note: if useCats=0 category labels are ignored as in proposal scoring.
    # Note: multiple areaRngs [Ax2] and maxDets [Mx1] can be specified.
    #
    # evaluate(): evaluates detections on every image and every category and
    # concats the results into the "evalImgs" with fields:
    #  dtIds      - [1xD] id for each of the D detections (dt)
    #  gtIds      - [1xG] id for each of the G ground truths (gt)
    #  dtMatches  - [TxD] matching gt id at each IoU or 0
    #  gtMatches  - [TxG] matching dt id at each IoU or 0
    #  dtScores   - [1xD] confidence of each dt
    #  gtIgnore   - [1xG] ignore flag for each gt
    #  dtIgnore   - [TxD] ignore flag for each dt at each IoU
    #
    # accumulate(): accumulates the per-image, per-category evaluation
    # results in "evalImgs" into the dictionary "eval" with fields:
    #  params     - parameters used for evaluation
    #  date       - date evaluation was performed
    #  counts     - [T,R,K,A,M] parameter dimensions (see above)
    #  precision  - [TxRxKxAxM] precision for every evaluation setting
    #  recall     - [TxKxAxM] max recall for every evaluation setting
    # Note: precision and recall==-1 for settings with no gt objects.
    #
    # See also coco, mask, pycocoDemo, pycocoEvalDemo
    #
    # Microsoft COCO Toolbox.      version 2.0
    # Data, paper, and tutorials available at:  http://mscoco.org/
    # Code written by Piotr Dollar and Tsung-Yi Lin, 2015.
    # Licensed under the Simplified BSD License [see coco/license.txt]
    def __init__(self, cocoGt=None, cocoDt=None, iouType='segm'):
        '''
        Initialize CocoEval using coco APIs for gt and dt
        :param cocoGt: coco object with ground truth annotations
        :param cocoDt: coco object with detection results
        :return: None
        '''
        if not iouType:
            print('iouType not specified. use default iouType segm')
        self.cocoGt   = cocoGt              # ground truth COCO API
        self.cocoDt   = cocoDt              # detections COCO API
        self.evalImgs = defaultdict(list)   # per-image per-category evaluation results [KxAxI] elements
        self.eval     = {}                  # accumulated evaluation results
        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation
        self.params = Params(iouType=iouType) # parameters
        self._paramsEval = {}               # parameters for evaluation
        self.stats = []                     # result summarization
        self.ious = {}                      # ious between all gts and dts
        if not cocoGt is None:
            # 把GT中所有的img id 与 类别 id 加入参数params 对应的 imgIds，catIds
            self.params.imgIds = sorted(cocoGt.getImgIds())
            self.params.catIds = sorted(cocoGt.getCatIds())


    def _prepare(self):
        '''
        Prepare ._gts and ._dts for evaluation based on params
        :return: None

        在目标检测中 _.gts 索引Ann的index为 [图片ip， 类别ip]，得到的是一个list数组，如果一张图片的一个类别有多个bbox，
        那么list中将会有多个item. _.dts同理
        '''
        def _toMask(anns, coco):
            # modify ann['segmentation'] by reference
            for ann in anns:
                rle = coco.annToRLE(ann)
                ann['segmentation'] = rle
        p = self.params
        if p.useCats:
            gts=self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
            dts=self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
        else:
            gts=self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds))
            dts=self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds))

        # convert ground truth to mask if iouType == 'segm'
        if p.iouType == 'segm':
            _toMask(gts, self.cocoGt)
            _toMask(dts, self.cocoDt)
        # set ignore flag
        # 设置忽略检测
        for gt in gts:
            gt['ignore'] = gt['ignore'] if 'ignore' in gt else 0
            gt['ignore'] = 'iscrowd' in gt and gt['iscrowd']
            if p.iouType == 'keypoints':
                gt['ignore'] = (gt['num_keypoints'] == 0) or gt['ignore']

        # 这种声明方式产生的self._gts是一个字典，字典的key是元祖(imgeid, catid), 字典的每个元素(value) 是列表
        # 这样得到的就是相同的img_id和类别id的信息，存放在一个列表中，即一张图像的同一个类别的所有框在一个列表中；
        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation
        # 给_gts字典对应的(imgeid, catid)， 添加对应的bbox信息
        # 得到的是每张图片，每个类别的检测结果的集合。
        for gt in gts:
            self._gts[gt['image_id'], gt['category_id']].append(gt)
        for dt in dts:
            self._dts[dt['image_id'], dt['category_id']].append(dt)
        self.evalImgs = defaultdict(list)   # per-image per-category evaluation results
        self.eval     = {}                  # accumulated evaluation results

    def evaluate(self):
        '''
        Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
        :return: None
        '''
        tic = time.time()
        print('Running per image evaluation...')
        p = self.params
        # add backward compatibility if useSegm is specified in params
        if not p.useSegm is None:
            p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
            print('useSegm (deprecated) is not None. Running {} evaluation'.format(p.iouType))
        print('Evaluate annotation type *{}*'.format(p.iouType))
        p.imgIds = list(np.unique(p.imgIds))  # 去重，得到唯一imgid
        if p.useCats:
            p.catIds = list(np.unique(p.catIds)) # 去重，得到唯一gt类别id，不包括背景
        p.maxDets = sorted(p.maxDets)
        self.params=p

        self._prepare()
        # loop through images, area range, max detection number
        catIds = p.catIds if p.useCats else [-1]

        # 根据检测人物选择 computeIoU 函数，这里可以自定义 computeIoU
        if p.iouType == 'segm' or p.iouType == 'bbox':
            computeIoU = self.computeIoU
        elif p.iouType == 'keypoints':
            computeIoU = self.computeOks

        # computeIoU 返回的是一个(M,N)的矩阵ndarry，
        # 表示一张图中某一个类别即(imgId, catId)的预测框(M个)和这个类别的gt(N个)的iou矩阵
        # 其中M是在这个(imgId, catId) 下有多少个预测的bbox，N是在这个(imgId, catId)下有多少个GT
        # self.ious 是一个字典，每个元素的value就是computeIoU返回的 (M,N)矩阵， key 是元祖 (imgId, catId)
        self.ious = {(imgId, catId): computeIoU(imgId, catId) \
                        for imgId in p.imgIds
                        for catId in catIds}

        evaluateImg = self.evaluateImg
        maxDet = p.maxDets[-1]

        # self.evalImgs是列表，每一个元素是字典，存储的是单张图片，一种类别，特定areaRng下的预测框dt和gt的匹配结果(在不同的阈值下)
        # 具体看 evaluateImg 函数
        self.evalImgs = [evaluateImg(imgId, catId, areaRng, maxDet)
                 for catId in catIds
                 for areaRng in p.areaRng
                 for imgId in p.imgIds
             ]
        self._paramsEval = copy.deepcopy(self.params)
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc-tic))

    # 这块用cython写的，主要返回的就是一张图片，一种类别即(imgId，catId)， 对应的M*N矩阵，每个值都是对应框的IoU值
    def computeIoU(self, imgId, catId):
        p = self.params
        if p.useCats:
            gt = self._gts[imgId,catId]
            dt = self._dts[imgId,catId]
        else:
            # 只有一类的情况，把这张图片的所有类别的所有检测结果进行一个数组的合并
            gt = [_ for cId in p.catIds for _ in self._gts[imgId,cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId,cId]]
        if len(gt) == 0 and len(dt) ==0:
            return []
        # 按照网络预测的置信度score排序， inds是score从大到小排列的索引
        inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in inds]

        # 将此处的dt按置信度从大到小排列
        # 其中，如果dt大于100个检测框的，按置信度取前100个(100个由p.maxDets设定)
        # 注意是一张图片一种类别的预测框不超过p.maxDets[-1]个，而不是一张图片的预测框不超过这么多，
        # 除非设置忽视类别，那就等价于一张图片的总的预测框不多于p.maxDets[-1]，因为本函数计算的是一张图片一个类别
        if len(dt) > p.maxDets[-1]:
            dt=dt[0:p.maxDets[-1]] # 把超出最大检测数量maxDets[-1]的bbox剔除

        if p.iouType == 'segm':
            g = [g['segmentation'] for g in gt]
            d = [d['segmentation'] for d in dt]
        elif p.iouType == 'bbox':
            g = [g['bbox'] for g in gt]
            d = [d['bbox'] for d in dt]
        else:
            raise Exception('unknown iouType for iou computation')

        # compute iou between each dt and gt region
        iscrowd = [int(o['iscrowd']) for o in gt]
        ious = maskUtils.iou(d,g,iscrowd)
        return ious

    def computeOks(self, imgId, catId):
        p = self.params
        # dimention here should be Nxm
        gts = self._gts[imgId, catId]
        dts = self._dts[imgId, catId]
        inds = np.argsort([-d['score'] for d in dts], kind='mergesort')
        dts = [dts[i] for i in inds]
        if len(dts) > p.maxDets[-1]:
            dts = dts[0:p.maxDets[-1]]
        # if len(gts) == 0 and len(dts) == 0:
        if len(gts) == 0 or len(dts) == 0:
            return []
        ious = np.zeros((len(dts), len(gts)))
        sigmas = p.kpt_oks_sigmas
        vars = (sigmas * 2)**2
        k = len(sigmas)
        # compute oks between each detection and ground truth object
        for j, gt in enumerate(gts):
            # create bounds for ignore regions(double the gt bbox)
            g = np.array(gt['keypoints'])
            xg = g[0::3]; yg = g[1::3]; vg = g[2::3]
            k1 = np.count_nonzero(vg > 0)
            bb = gt['bbox']
            x0 = bb[0] - bb[2]; x1 = bb[0] + bb[2] * 2
            y0 = bb[1] - bb[3]; y1 = bb[1] + bb[3] * 2
            for i, dt in enumerate(dts):
                d = np.array(dt['keypoints'])
                xd = d[0::3]; yd = d[1::3]
                if k1>0:
                    # measure the per-keypoint distance if keypoints visible
                    dx = xd - xg
                    dy = yd - yg
                else:
                    # measure minimum distance to keypoints in (x0,y0) & (x1,y1)
                    z = np.zeros((k))
                    dx = np.max((z, x0-xd),axis=0)+np.max((z, xd-x1),axis=0)
                    dy = np.max((z, y0-yd),axis=0)+np.max((z, yd-y1),axis=0)
                e = (dx**2 + dy**2) / vars / (gt['area']+np.spacing(1)) / 2
                if k1 > 0:
                    e=e[vg > 0]
                ious[i, j] = np.sum(np.exp(-e)) / e.shape[0]
        return ious

    def evaluateImg(self, imgId, catId, aRng, maxDet):
        '''
        perform evaluation for single category and image
        :return: dict (single image results)
        计算本张图片(imgId)，特定类别(catId)，特定面积阈值(aRng)，特定最大检测结果(maxDet)下的result。
        '''
        p = self.params
        if p.useCats:
            # 本张图片,特定类别的 gt 和 所有检测结果dt
            gt = self._gts[imgId,catId]
            dt = self._dts[imgId,catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId,cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId,cId]]
        if len(gt) == 0 and len(dt) ==0:
            return None

        for g in gt:
            # 如果 gt 不符合特定面积的阈值，就忽略，设置ignore
            if g['ignore'] or (g['area']<aRng[0] or g['area']>aRng[1]):
                g['_ignore'] = 1  # 忽略的值为 1
            else:
                g['_ignore'] = 0

        # sort dt highest score first, sort gt ignore last
        # gt 按 g['_ignore'] 值排序
        # gtind 前面都是 ignore为0 的gt(不忽略)， 后面都是 ignore为1的gt
        gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
        gt = [gt[i] for i in gtind]   # 得到按ignore 排序后的 gt

        # 按score 从大到小排序，挑出满足这个最大检测个数maxDet下的所有dt
        dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in dtind[0:maxDet]]
        iscrowd = [int(o['iscrowd']) for o in gt]
        # load computed ious

        # 一张图片中，一种类别的gt存在，按gtind取gt与所有dt的iou结果
        ious = self.ious[imgId, catId][:, gtind] if len(self.ious[imgId, catId]) > 0 else self.ious[imgId, catId]

        T = len(p.iouThrs)  # 设置最后的阈值卡关
        G = len(gt)         # 总的gt数量，包括要忽略的
        D = len(dt)         # 总的dt数量

        # 在每个阈值下的gt是否得到匹配
        # 存储的是每一个iou阈值下的gt能够匹配到的最大iou对应的模型预测框dt(最多p.maxDet[-1]个)的id，匹配不到的值是0；
        # gtm存储的是匹配成功的那个gt对应的dt所对应的id,
        gtm  = np.zeros((T,G))

        # 在每个阈值下的Dt是否得到匹配
        # 存储的是每一个iou阈值下的模型预测框dt匹配到的gt的id，匹配不到的是0；
        dtm  = np.zeros((T,D))

        gtIg = np.array([g['_ignore'] for g in gt])  # 所有忽略的gt
        # 所有忽略的dt，表示每一个阈值下的预测框dt匹配到的gt是否需要ignore
        dtIg = np.zeros((T,D))

        # dt已经按照置信度排过序，gt已经按照ignore排过位置，非ignore在前，ignore在后面
        # 下面的if里面实现的功能是每一个iou阈值下，遍历预测框(预测框已经按置信度从大到小排序)，如果一个预测框dt和gt匹配上，
        # 则另一个预测框不能再通过iou和这个gt进行匹配
        if not len(ious)==0:    # 如果这张图片存在这个类别的gt与dt， 此时的ious仅仅是 不忽略的那些gt对应的样本
            for tind, t in enumerate(p.iouThrs):  # tind, t 是 IoU index，IoU阈值
                # dt按照置信度大小排序好的前 max_Det个d
                for dind, d in enumerate(dt):
                    # information about best match so far (m=-1 -> unmatched)
                    iou = min([t,1-1e-10])
                    # 如果m= -1 代表这个dt没有得到匹配, m代表dt匹配的最好的gt的索引下标
                    m   = -1
                    for gind, g in enumerate(gt):  #  在gt和dt之间进行遍历
                        # if this gt already matched, and not a crowd, continue
                        # 如果该阈值下的这个gt已经被其他置信度更好的dt匹配到了，本轮的dt就不能匹配这个gt了，直接下个gt计算是否匹配
                        if gtm[tind,gind]>0 and not iscrowd[gind]:
                            continue
                        # if dt matched to reg gt, and on ignore gt, stop
                        # 因为gt已经按照ignore排好序了，前面的为0，后面的为1. 于是当我们碰到第一个gt的ignore为1时(gtIg[gind]==1)，
                        # 判断这个dt是否已经匹配到了其他的gt，因为gind之后的都被ignore了
                        # 如果m>-1，并且m对应的gt没有被ignore（gtIg[m]==0），就直接结束即可，对应的就是这个dt匹配最好的gt
                        # 跳出循环条件？等待已经匹配成功，m此时一定是正数，并且匹配的这个不是ignore,
                        # 并且下一个就是ignore，这样才跳出循环，这样保证会一直寻找最优的配置
                        if m>-1 and gtIg[m]==0 and gtIg[gind]==1:
                            break
                        # continue to next gt unless better match made
                        # 如果计算dt与gt的iou小于目前最佳的IoU，忽略这个gt，计算下一个gt
                        if ious[dind,gind] < iou:
                            continue
                        # if match successful and best so far, store appropriately
                        # 此时表示匹配成功了，超过当前最佳的IoU， m存储的是匹配成功的gtind，
                        # 先更新IoU与m的值，但是还是会继续，是为了找到更好的匹配者
                        iou=ious[dind,gind]
                        m=gind
                    # if match made store id of match for both dt and gt
                    # 如果这个dt没有对应的gt与其匹配，继续dt的下一个循环
                    if m ==-1:
                        continue

                    # 此时m != -1,如果这个dt有对应的gt与其匹配
                    # 把当前dt与第m个gt进行匹配，修改dtm与gtm的值，分别一一对应

                    # 对应的能匹配上gt的预测框是否需要ignore，
                    # 如果这个dt对应的最佳gt本身就是被ignore的，就把这个dt也设置为ignore
                    dtIg[tind,dind] = gtIg[m]        # dind对应的那个gt是不是ignore
                    dtm[tind,dind]  = gt[m]['id']    # dt匹配上的gt的id
                    gtm[tind,m]     = d['id']        # gt中的框匹配上的预测框dt的id，也就是本次dind对应的id
        # set unmatched detections outside of area range to ignore
        # dtm中没有匹配到gt的预测框，同时预测框的area在指定的aRng范围外的，则设置dtIg中对应的预测框为ignore
        # 有哪些dt需要ignore呢？首先匹配上的框本身即是ignore的，另外没有匹配到的在面积范围之外的也要ignore
        # 匹配到gt的预测框dt， 但是dt在aRng范围外正常计算，不ignore
        a = np.array([d['area']<aRng[0] or d['area']>aRng[1] for d in dt]).reshape((1, len(dt)))
        dtIg = np.logical_or(dtIg, np.logical_and(dtm==0, np.repeat(a,T,0)))
        # store results for given image and category
        return {
                'image_id':     imgId,
                'category_id':  catId,
                'aRng':         aRng,
                'maxDet':       maxDet,                    # 这里是p.maxDets[-1]
                'dtIds':        [d['id'] for d in dt],     # 已经排过序的预测框id
                'gtIds':        [g['id'] for g in gt],
                'dtMatches':    dtm,                       # (T,D) 其中D是已经按置信度排除的bbox
                'gtMatches':    gtm,                       # (T,G) G是按照aRng等信息排序的不ignore在前，ignore在后的gt
                'dtScores':     [d['score'] for d in dt],  # 已经排过序的score
                'gtIgnore':     gtIg,                      # G指的是单张图片特定aRng的gt是否ignore信息
                'dtIgnore':     dtIg,                      # (T,D)
            }

    def accumulate(self, p = None):
        '''
        Accumulate per image evaluation results and store the result in self.eval
        :param p: input params for evaluation
        :return: None
        '''
        print('Accumulating evaluation results...')
        tic = time.time()
        if not self.evalImgs:
            print('Please run evaluate() first')
        # allows input customized parameters
        if p is None:
            p = self.params
        p.catIds = p.catIds if p.useCats == 1 else [-1]
        T           = len(p.iouThrs)   # iou阈值的个数
        R           = len(p.recThrs)   # 召回recThrs阈值的个数
        K           = len(p.catIds) if p.useCats else 1  # 多少个类
        A           = len(p.areaRng)   # 多少个面积阈值
        M           = len(p.maxDets)   # 多少个最大检测数
        precision   = -np.ones((T,R,K,A,M)) # -1 for the precision of absent categories
        recall      = -np.ones((T,K,A,M))
        scores      = -np.ones((T,R,K,A,M))

        # create dictionary for future indexing
        _pe = self._paramsEval
        catIds = _pe.catIds if _pe.useCats else [-1]
        setK = set(catIds)
        setA = set(map(tuple, _pe.areaRng))
        setM = set(_pe.maxDets)
        setI = set(_pe.imgIds)
        # get inds to evaluate
        k_list = [n for n, k in enumerate(p.catIds)  if k in setK]  # 对应不重复的K的id list, 下同
        m_list = [m for n, m in enumerate(p.maxDets) if m in setM]  # 对应不重复的maxDets的值list
        a_list = [n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng)) if a in setA]
        i_list = [n for n, i in enumerate(p.imgIds)  if i in setI]
        I0 = len(_pe.imgIds)  # 多少个图片
        A0 = len(_pe.areaRng) # 多少个面积阈值

        # 根据self.evalImgs的存储形式，遍历时最里层是img_id(i_list)、次外层是aRng(a_list)、最外层是类别id(k_list)
        '''
            self.evalImgs = [evaluateImg(imgId, catId, areaRng, maxDet)
                for catId in catIds
                for areaRng in p.areaRng
                for imgId in p.imgIds
            ]
        '''
        # retrieve E at each category, area range, and max number of detections
        for k, k0 in enumerate(k_list):      # 类别的索引下标遍历
            Nk = k0*A0*I0                    # 当前K0前面过了多少图片与面积阈值
            for a, a0 in enumerate(a_list):  # aRng的遍历
                Na = a0*I0                   # 在当前K0前面过了多少阈值
                for m, maxDet in enumerate(m_list):
                    E = [self.evalImgs[Nk + Na + i] for i in i_list]  # k0，a0下的所有Images(i_list)
                    E = [e for e in E if not e is None]  # 等价于  if e is not None
                    if len(E) == 0:
                        continue

                    # 特定类别、特定aRng的所有图片中，每一张图片的maxDet个预测框得分,即
                    # k0，a0，maxdet下所有Images得分的索引 inds
                    dtScores = np.concatenate([e['dtScores'][0:maxDet] for e in E])

                    # different sorting method generates slightly different results.
                    # mergesort is used to be consistent as Matlab implementation.

                    # 是将特定类别，特定aRng的所有图片的预测框
                    # (每张图片特定类别、aRng取置信度从大到小的maxDet个框)
                    # 按照得分从高到低排序
                    inds = np.argsort(-dtScores, kind='mergesort')
                    dtScoresSorted = dtScores[inds]

                    # dtm、dtIg维度是(T,maxDet个数*图片个数)
                    # 在当前k0,a0下，每张图片不超过MaxDet的所有det按照ind排序。 dtm[T,sum(Det) in every imges]
                    dtm  = np.concatenate([e['dtMatches'][:,0:maxDet] for e in E], axis=1)[:,inds]
                    dtIg = np.concatenate([e['dtIgnore'][:,0:maxDet]  for e in E], axis=1)[:,inds]
                    gtIg = np.concatenate([e['gtIgnore'] for e in E])   # gtIg维度是(图片个数，G)
                    npig = np.count_nonzero(gtIg==0 )       # gt不ignore的个数
                    # 有多少个正样本
                    if npig == 0:
                        continue

                    # 如果dtm对应的匹配gt不为0，且对应的gt没有被忽略，这个dt就是TP
                    tps = np.logical_and(               dtm,  np.logical_not(dtIg) )

                    # dtm对应的gt为0， 并且这个dt也没有被忽略，这个dt就是FP
                    fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg) )

                    # 按照行的方式（每个Iou阈值下）进行匹配到的累加 每个index也就是到这个置信度的时候有多少个tp，有多少个fp
                    tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
                    fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)

                    # 遍历所有的tp和fp样本
                    for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                        tp = np.array(tp)  # 得到这个Iou下对应的tp
                        fp = np.array(fp)  # 得到这个IoU下对应的fp
                        nd = len(tp)       # 有多少个tp
                        rc = tp / npig     # 每个置信度分数下对应的recall
                        pr = tp / (fp+tp+np.spacing(1))  #每个阶段对应的精度precision
                        q  = np.zeros((R,))  # 特定召回率下的precision值(pr曲线)
                        ss = np.zeros((R,))  # 特定召回率下的对应的bbox的置信度

                        if nd:
                            recall[t,k,a,m] = rc[-1]  #     recall的话取最后一个
                        else:
                            recall[t,k,a,m] = 0

                        # numpy is slow without cython optimization for accessing elements
                        # use python array gets significant speed improvement
                        pr = pr.tolist(); q = q.tolist()

                        # 当前i下的最大精度
                        for i in range(nd-1, 0, -1):
                            if pr[i] > pr[i-1]:
                                pr[i-1] = pr[i]

                        # 找到每个recall发生变化的时候的index，与p.recThrs一一对应，最接近其的值的index
                        # 这里调用的np.searchsorted表示p.recThrs中每一个值能插入到rc中的位置索引，其中rc必须是升序
                        inds = np.searchsorted(rc, p.recThrs, side='left')
                        try:
                            for ri, pi in enumerate(inds):
                                q[ri] = pr[pi]      # 得到每个recall阈值对应的最大精度，存入q中
                                ss[ri] = dtScoresSorted[pi]  # 得到这个recall值下的得分
                        except:
                            pass
                        precision[t,:,k,a,m] = np.array(q)  # 按照recall的大小存入对应的精度
                        scores[t,:,k,a,m] = np.array(ss)    # 存入对应的分数
        self.eval = {
            'params': p,
            'counts': [T, R, K, A, M],
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'precision': precision,
            'recall':   recall,
            'scores': scores,
        }
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format( toc-tic))

    def summarize(self):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''
        def _summarize( ap=1, iouThr=None, areaRng='all', maxDets=100 ):
            p = self.params
            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap==1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            # 如果是'all' 就是所有尺度， 如果不是就是特定的尺度
            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]

            # 如果是ap，就从precision中得到对应面积阈值、最大检测数下的精度
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                # IoU
                # 得到特定IoU下的所有pr
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]  # 找到对应的threshold的得分，如果在输入参数里面制定了阈值范围
                    s = s[t]
                s = s[:,:,:,aind,mind]
            else:
                # 如果是recall，就取出recall的值
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:,aind,mind]
            if len(s[s>-1])==0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s>-1])  #除去-1 其他的计算平均精度
            print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
            return mean_s
        def _summarizeDets():
            stats = np.zeros((12,))
            # all iouThr， 所有recall下，所有面积下， 所有类别，在最大检测数100下的的平均AP，下同
            stats[0] = _summarize(1)
            stats[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
            stats[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
            stats[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
            stats[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
            # all iouThr，所有面积下， 所有类别，在最大检测数1下的的平均recall, 下同
            stats[6] = _summarize(0, maxDets=self.params.maxDets[0])
            stats[7] = _summarize(0, maxDets=self.params.maxDets[1])
            stats[8] = _summarize(0, maxDets=self.params.maxDets[2])
            stats[9] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
            stats[10] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[11] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])
            return stats
        def _summarizeKps():
            stats = np.zeros((10,))
            stats[0] = _summarize(1, maxDets=20)
            stats[1] = _summarize(1, maxDets=20, iouThr=.5)
            stats[2] = _summarize(1, maxDets=20, iouThr=.75)
            stats[3] = _summarize(1, maxDets=20, areaRng='medium')
            stats[4] = _summarize(1, maxDets=20, areaRng='large')
            stats[5] = _summarize(0, maxDets=20)
            stats[6] = _summarize(0, maxDets=20, iouThr=.5)
            stats[7] = _summarize(0, maxDets=20, iouThr=.75)
            stats[8] = _summarize(0, maxDets=20, areaRng='medium')
            stats[9] = _summarize(0, maxDets=20, areaRng='large')
            return stats
        if not self.eval:
            raise Exception('Please run accumulate() first')
        iouType = self.params.iouType
        if iouType == 'segm' or iouType == 'bbox':
            summarize = _summarizeDets
        elif iouType == 'keypoints':
            summarize = _summarizeKps
        self.stats = summarize()

    def __str__(self):
        self.summarize()

class Params:
    '''
    Params for coco evaluation api
    '''
    def setDetParams(self):
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        self.recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
        self.maxDets = [1, 10, 100]
        self.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        self.areaRngLbl = ['all', 'small', 'medium', 'large']
        self.useCats = 1

    def setKpParams(self):
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        self.recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
        self.maxDets = [20]
        self.areaRng = [[0 ** 2, 1e5 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        self.areaRngLbl = ['all', 'medium', 'large']
        self.useCats = 1
        self.kpt_oks_sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89])/10.0

    def __init__(self, iouType='segm'):
        if iouType == 'segm' or iouType == 'bbox':
            self.setDetParams()
        elif iouType == 'keypoints':
            self.setKpParams()
        else:
            raise Exception('iouType not supported')
        self.iouType = iouType
        # useSegm is deprecated
        self.useSegm = None
