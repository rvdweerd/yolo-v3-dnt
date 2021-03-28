"""
Main file for training Yolo model on Pascal VOC and COCO dataset
"""

import config
import torch
import torch.optim as optim

from model import YOLOv3
from tqdm import tqdm
from utils import (
    mean_average_precision,
    cells_to_bboxes,
    get_evaluation_bboxes,
    get_evaluation_bboxes_darknet,
    save_checkpoint,
    load_checkpoint,
    check_class_accuracy,
    get_loaders,
    plot_couple_examples,
    plot_image,
    plot_bboxes_on_img
)
from loss import YoloLoss
import cv2
import numpy as np


torch.backends.cudnn.benchmark = True

def load_dn():
	#net = cv2.dnn.readNet("~/darknet_training/doubleclass/backup/tyv3_94-91.weights", "~/darknet_training/doubleclass/cfg/yolov3-tiny_3l.cfg")
    #net = cv2.dnn.readNetFromDarknet("~/darknet_training/doubleclass/backup/yolov3-tiny_3l_final.weights", "~/darknet_training/doubleclass/cfg/yolov3-tiny_3l.cfg")
    net = cv2.dnn.readNetFromDarknet("../darknet_training/doubleclass/cfg/yolov3-tiny_3l.cfg","../darknet_training/doubleclass/backup/yolov3-tiny_3l_best.weights")
    #net = cv2.dnn.readNetFromDarknet("../darknet_training/doubleclass/cfg/yolov3-tiny_3l.cfg","../darknet_training/doubleclass/backup/tyv3_94-91.weights")
    #classes = ['ball','robot']
    classes = [0,1]
    layers_names = net.getLayerNames()
    output_layers = [layers_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    return net, classes, colors, output_layers




def evaluate_fn(eval_loader, model, scaled_anchors):
    loop = tqdm(train_loader, leave=True)
    for batch_idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)
        y0, y1, y2 = (
            y[0].to(config.DEVICE),
            y[1].to(config.DEVICE),
            y[2].to(config.DEVICE),
        )
        
        model.evaluate()
        
        with torch.cuda.amp.autocast():
            out = model(x)
        


def main():
    model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    dnmodel,_,_,outlayers = load_dn()


    train_loader, test_loader, eval_loader = get_loaders(
        #train_csv_path=config.DATASET + "/train.csv", test_csv_path=config.DATASET + "/test.csv"
        train_csv_path=config.DATASET + "/train.csv", test_csv_path=config.DATASET + "/test.csv"#examples_small.csv"
    )
    # load checkpoint
    print("=> Loading checkpoint")
    checkpoint = torch.load(config.CHECKPOINT_FILE, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])

    scaled_anchors = (
        torch.tensor(config.ANCHORS)
        * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(config.DEVICE)
    print(config.DEVICE)

    darknet_eval=True
    if darknet_eval:
        pred_boxes_darknet, true_boxes_darknet = get_evaluation_bboxes_darknet(
                    eval_loader,
                    dnmodel,
                    outlayers,
                    iou_threshold=config.NMS_IOU_THRESH,
                    anchors=config.ANCHORS,
                    threshold=config.CONF_THRESHOLD,
                )


    pred_boxes, true_boxes = get_evaluation_bboxes(
                eval_loader,
                model,
                iou_threshold=config.NMS_IOU_THRESH,
                anchors=config.ANCHORS,
                threshold=config.CONF_THRESHOLD,
            )
    images_bbox_map={}
    for i in range(eval_loader.__sizeof__()*eval_loader.__len__()):
        images_bbox_map[i]={'gt_bboxes':[],'pred_bboxes':[]}
    for pbox in pred_boxes:
        images_bbox_map[pbox[0]]['pred_bboxes'].append(pbox[1:])
    for tbox in true_boxes:
        images_bbox_map[tbox[0]]['gt_bboxes'].append(tbox[1:])

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video=cv2.VideoWriter('video2.avi', fourcc, 2, (640,480))

    img_count=0
    for batch_idx, (x, labels, im_name) in enumerate(tqdm(eval_loader)):
        for i in range(x.shape[0]):
            img=plot_bboxes_on_img(x[i].permute(1,2,0),images_bbox_map[img_count],im_name[i])
            video.write(img)
            img_count+=1
    #cv2.destroyAllWindows()
    video.release()

if __name__ == "__main__":
    main()
