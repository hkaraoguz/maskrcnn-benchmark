from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
import cv2
config_file = "../configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml"

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cpu"])

coco_demo = COCODemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.7,
)
# load image and then run prediction
image = cv2.imread('test.jpg') 
resultimg,boxes = coco_demo.run_on_opencv_image(image)
#print(dir(boxes))
#print(boxes)
print(boxes.bbox)
print(boxes.get_field("labels"))
print(boxes.get_field("scores"))

labels = boxes.get_field("labels")

for label in labels:
    print(coco_demo.CATEGORIES[label])
#print(dir(boxes.bbox))
cv2.imshow("res",resultimg)
cv2.waitKey(0)
