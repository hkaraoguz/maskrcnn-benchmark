from maskrcnn_benchmark.config import cfg
from maskrcnn.demo.predictor import COCODemo
import cv2
config_file = "/home/cybercomml/Dev/CyberCV/maskrcnn/configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml"

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cpu"])

coco_demo = COCODemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.7,
)

def detect(imgs):
  img_results = []
  for img in imgs:
    resultimg,boxes = coco_demo.run_on_opencv_image(img)
    labels = boxes.get_field("labels")
    scores = boxes.get_field("scores")
    result_labels = []
    result_scores = []
    for i,label in enumerate(labels):
       result_labels.append(coco_demo.CATEGORIES[label])
       result_scores.append(scores[i].item())
    current_result = []
    boxs = boxes.bbox
    print(result_scores)
    print(result_labels)
    for i,box in enumerate(boxs):
       #bbox = box.to(torch.int64)
       #print(box)
       bbox = box.tolist()
       #print(bbox)
       #print(result_scores[i])
       #print(result_labels[i])
       current_result.append([bbox[0],bbox[1],bbox[2],bbox[3],float(result_scores[i]),result_labels[i]])
    img_results.append(current_result)
   
  return img_results
      

# load image and then run prediction
#image = cv2.imread('test.jpg') 

#print(dir(boxes))
#print(boxes)
#print(boxes.bbox)
#print(boxes.get_field("labels"))
#print(boxes.get_field("scores"))


#print(dir(boxes.bbox))
#cv2.imshow("res",resultimg)
#cv2.waitKey(0)
