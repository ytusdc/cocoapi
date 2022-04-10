##The code of eval_coco.py
import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
from cocoeval_self import COCOeval_self
results = r'./jsonfile/my_result.json'  ##模型预测结果
anno = r'./jsonfile/instances_val2017.json'  ##ground truth

coco_gt = coco.COCO(anno)
coco_dets = coco_gt.loadRes(results)
coco_eval = COCOeval_self(coco_gt, coco_dets, "bbox")
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()
coco_eval.get_good_predict_data()
