import config
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import random
import torch

from collections import Counter
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2
from PIL import Image

def iou_width_height(boxes1, boxes2):
    """
    Parameters:
        boxes1 (tensor): width and height of the first bounding boxes
        boxes2 (tensor): width and height of the second bounding boxes
    Returns:
        tensor: Intersection over union of the corresponding boxes
    """
    intersection = torch.min(boxes1[..., 0].to(torch.float32), boxes2[..., 0]) * torch.min(
        boxes1[..., 1].to(torch.float32), boxes2[..., 1]
    )
    union = (
        boxes1[..., 0].to(torch.float32) * boxes1[..., 1].to(torch.float32) + boxes2[..., 0] * boxes2[..., 1] - intersection
    )
    return intersection / union


def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Video explanation of this function:
    https://youtu.be/XXYG5ZWtjj0

    This function calculates intersection over union (iou) given pred boxes
    and target boxes.

    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)

    Returns:
        tensor: Intersection over union for all examples
    """

    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    if box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


def non_max_suppression(bboxes, iou_threshold, threshold, box_format="corners"):
    """
    Video explanation of this function:
    https://youtu.be/YDkjWEN8jNA

    Does Non Max Suppression given bboxes

    Parameters:
        bboxes (list): list of lists containing all bboxes with each bboxes
        specified as [class_pred, prob_score, x1, y1, x2, y2]
        iou_threshold (float): threshold where predicted bboxes is correct
        threshold (float): threshold to remove predicted bboxes (independent of IoU)
        box_format (str): "midpoint" or "corners" used to specify bboxes

    Returns:
        list: bboxes after performing NMS given a specific IoU threshold
    """

    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
            or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format,
            )
            < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms


def mean_average_precision(
    pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=20
):
    """
    Video explanation of this function:
    https://youtu.be/FppOzcDvaDI

    This function calculates mean average precision (mAP)

    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar as pred_boxes except all the correct ones
        iou_threshold (float): threshold where predicted bboxes is correct
        box_format (str): "midpoint" or "corners" used to specify bboxes
        num_classes (int): number of classes

    Returns:
        float: mAP value across all classes given a specific IoU threshold
    """

    # list storing all AP for respective classes
    average_precisions = []

    # used for numerical stability later on
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []

        # Go through all predictions and targets,
        # and only add the ones that belong to the
        # current class c
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # find the amount of bboxes for each training example
        # Counter here finds how many ground truth bboxes we get
        # for each training example, so let's say img 0 has 3,
        # img 1 has 5 then we will obtain a dictionary with:
        # amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        # We then go through each key, val in this dictionary
        # and convert to the following (w.r.t same example):
        # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # sort by box probabilities which is index 2
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        # If none exists for this class then we can safely skip
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same
            # training idx as detection
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format,
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                # only detect ground truth detection once
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    # true positive and add this bounding box to seen
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            # if IOU is lower then the detection is a false positive
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        # torch.trapz for numerical integration
        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)


def plot_image(image, boxes, save_img=False):
    """Plots predicted bounding boxes on the image"""
    cmap = plt.get_cmap("tab20b")
    class_labels = config.PASCAL_CLASSES if config.DATASET=='PASCAL_VOC' else config.DNT_2_CLASSES
    colors = [cmap(i) for i in np.linspace(0, 1, len(class_labels))]
    im = np.array(image)
    height, width, _ = im.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)

    # box[0] is x midpoint, box[2] is width
    # box[1] is y midpoint, box[3] is height

    # Create a Rectangle patch
    for box in boxes:
        assert len(box) == 6, "box should contain class pred, confidence, x, y, width, height"
        class_pred = box[0]
        box = box[2:]
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=2,
            edgecolor=colors[int(class_pred)],
            facecolor="none",
        )
        # Add the patch to the Axes
        ax.add_patch(rect)
        plt.text(
            upper_left_x * width,
            upper_left_y * height,
            s=class_labels[int(class_pred)],
            color="white",
            verticalalignment="top",
            bbox={"color": colors[int(class_pred)], "pad": 0},
        )
    if save_img:
        plt.savefig("test.png")
    else:
        plt.show()

#from matplotlib.backends.backend_agg import FigureCanvasAgg
#from matplotlib.figure import Figure
def plot_bboxes_on_img(image, bbox_map, im_name):
    class_labels = config.PASCAL_CLASSES if config.DATASET=='PASCAL_VOC' else config.DNT_2_CLASSES

    #cmap = plt.get_cmap("tab20b")
    #colors = [cmap(i) for i in np.linspace(0, 1, len(class_labels))]
    colors=["red","blue"]

    im = np.array(image)
    height, width, _ = im.shape
    fig, ax = plt.subplots(1)
    #canvas = FigureCanvasAgg(fig)
    # Display the image
    ax.imshow(im)
    #plt.savefig("tmp.png")
    
    #cv2.imshow('img',im)
    #cv2.waitkey(0)
    #cv2.destroyAllWindows()

    #im=cv2.imread("tmp.png")
    #cv2.imwrite("test2.png",im)

    # Create a gt patches
    for box in bbox_map['gt_bboxes']:
        assert len(box) == 6, "box should contain class pred, confidence, x, y, width, height"
        class_pred = box[0]
        box = box[2:]
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=1,
            #edgecolor=colors[int(class_pred)],
            edgecolor=(1.,1.,1.,1.),
            facecolor="none",
        )
        # Add the patch to the Axes
        ax.add_patch(rect)
        #plt.text(
        #    upper_left_x * width,
        #    upper_left_y * height,
        #    s=class_labels[int(class_pred)],
        #    color="white",
        #    verticalalignment="top",
        #    bbox={"color": colors[int(class_pred)], "pad": 0},
        #)
    
    # Create a prediction patches
    for box in bbox_map['pred_bboxes']:
        assert len(box) == 6, "box should contain class pred, confidence, x, y, width, height"
        class_pred = box[0]
        box = box[2:]
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=1,
            edgecolor=colors[int(class_pred)],#"red",#colors[int(class_pred)],
            #edgecolor=(1.,1.,1.,1.),
            facecolor="none",
        )
        # Add the patch to the Axes
        ax.add_patch(rect)
        #plt.text(
        #    upper_left_x * width,
        #    upper_left_y * height,
        #    s=class_labels[int(class_pred)],
        #    color="white",
        #    verticalalignment="top",
        #    bbox={"color": colors[int(class_pred)], "pad": 0},
        #)
    plt.text(
        10,20,im_name,color="white"
    )
    rect_gt    = patches.Rectangle( (10,30),10,10, linewidth=1, edgecolor="white",facecolor="none")
    rect_ball  = patches.Rectangle( (10,373),10,10, linewidth=1, edgecolor="red",facecolor="none")
    rect_robot = patches.Rectangle( (10,393),10,10, linewidth=1, edgecolor="blue",facecolor="none")
    ax.add_patch(rect_gt)
    ax.add_patch(rect_ball)
    ax.add_patch(rect_robot)
    plt.text( 25,40, "= Ground Truth",color="white")
    plt.text( 25,383,"= Predicted Ball",color="white")
    plt.text( 25,403,"= Predicted Robot",color="white")

    plt.savefig("tmp.png")
    plt.close()
    #canvas.draw()
    #buf=canvas.buffer_rgba()
    #x=np.asarray(buf)
    x=cv2.imread("tmp.png")
    return x
    
def get_evaluation_bboxes(
    loader,
    model,
    iou_threshold,
    anchors,
    threshold,
    box_format="midpoint",
    device="cuda",
):
    # make sure model is in eval before get bboxes
    model.eval()
    train_idx = 0
    all_pred_boxes = []
    all_true_boxes = []
    for batch_idx, (x, labels, im_name) in enumerate(tqdm(loader)):
        x = x.to(device)
        #print(im_name)

        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]
        bboxes = [[] for _ in range(batch_size)]
        for i in range(3):
            S = predictions[i].shape[2]
            anchor = torch.tensor([*anchors[i]]).to(device) * S
            boxes_scale_i = cells_to_bboxes(
                predictions[i], anchor, S=S, is_preds=True
            )
            for idx, (box) in enumerate(boxes_scale_i):
                bboxes[idx] += box

        # we just want one bbox for each label, not one for each scale
        true_bboxes = cells_to_bboxes(
            labels[2], anchor, S=S, is_preds=False
        )

        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                bboxes[idx],
                iou_threshold=iou_threshold,
                threshold=threshold,
                box_format=box_format,
            )

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1

    model.train()
    return all_pred_boxes, all_true_boxes

import time

def get_evaluation_bboxes_darknet(
    loader,
    model,
    outputLayers,
    iou_threshold,
    anchors,
    threshold,
    box_format="midpoint",
    device="cuda",
):
    # make sure model is in eval before get bboxes
    #model.eval()
    classes=['ball','robot']
    colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')
    train_idx = 0
    all_pred_boxes = []
    all_true_boxes = []
    model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

    # examples=[
    #     "frame_a_910.png","01_06_2018__17_17_01_0000_upper.png","frame_a_1070.png","frame_a_2810.png","frame_a_3040.png",
    #     "frame_a_3710.png","frame_a_8240.png","frame_b_1880.png","image12.jpg",
    #     "image21.jpg","image113.jpg","image50167.jpg","image90178.jpg"
    #     ]
    examples=[]
    file_obj=open("DNT_2/train.csv")
    lines=file_obj.readlines()
    for line in lines[1:]:
       a,b=line.split(",")
       examples.append(a)
    examples.sort()

    def trackbar2(x):
        confidence = x/100
        r = r0.copy()
        for output in np.vstack(outputs):
            if output[4] > confidence:
                x, y, w, h = output[:4]
                p0 = int((x-w/2)*416), int((y-h/2)*416)
                p1 = int((x+w/2)*416), int((y+h/2)*416)
                cv2.rectangle(r, p0, p1, 1, 1)
        cv2.imshow('blob', r)
        text = f'Bbox confidence={confidence}'
        cv2.displayOverlay('blob', text)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    if not os.path.isdir('test_examples'):
        os.mkdir('test_examples')
    show_pipeline=True
    record=False
    if record:
        VID_w=832#640
        VID_h=728#480
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video=cv2.VideoWriter('test_examples/video.avi', fourcc, 2,(VID_w,VID_h))
    
    for ex in (examples[3:]):
        #img=cv2.imread("DNT_2/images/frame_a_2760.png")
        img=cv2.imread("DNT_2/images/"+ex)

        # Using PIL (https://github.com/AlexeyAB/darknet/issues/3119#issuecomment-506673120)
        #img=Image.open("DNT_2/images/"+ex)
        #img=img.resize((416,416),Image.BICUBIC).convert('RGB')


        if show_pipeline:
            #print('step1: img size',img.shape)
            cv2.imshow('window',  img)
            #cv2.imwrite('test_examples/ex'+str(idx)+'-1_orig_-'+ex,img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        #img = cv2.resize(img, None, fx=416/640, fy=416/640)
        #img = cv2.resize(img, None, fx=416/832, fy=416/832)
        #img=cv2.copyMakeBorder(img,0,416-312,0,0,cv2.BORDER_CONSTANT)
        
        #h,w,c=img.shape
        #img=cv2.resize(img,None,fx=416/w,fy=416/h,interpolation=cv2.INTER_CUBIC)
        
        
        img=cv2.resize(img,(416,416),interpolation=cv2.INTER_CUBIC)
        blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416),(0,0,0),swapRB=True, crop=False)
        r=blob[0,0,:,:]
        text = f'Blob shape={blob.shape}'
        if show_pipeline:
            print('step2: blob size',r.shape)
            cv2.imshow('blob',  r)
            cv2.displayOverlay('blob', text)
            #cv2.imwrite('test_examples/ex'+str(idx)+'-2_blob_-'+ex,img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        print(ex,'blob size',blob.shape, 'avg',blob.mean(),'max',blob.max())
        model.setInput(blob)
        outputs=model.forward(outputLayers)

        r0 = blob[0, 0, :, :]
        r = r0.copy()
        if show_pipeline:
            print('step3: blob size',r.shape)
            cv2.imshow('blob', r)
            cv2.createTrackbar('confidence', 'blob', 50, 101, trackbar2)
            trackbar2(50)

        boxes = []
        confidences = []
        classIDs = []
        h, w = img.shape[:2]

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if confidence > 0.5:
                    box = detection[:4] * np.array([w, h, w, h])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    box = [x, y, int(width), int(height)]
                    boxes.append(box)
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        if len(indices) > 0:
            for i in indices.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                color = [int(c) for c in colors[classIDs[i]]]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(classes[classIDs[i]], confidences[i])
                cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        if record:
            h,w,c=img.shape
            img_=cv2.copyMakeBorder(img,0,VID_h-h,0,VID_w-w,cv2.BORDER_CONSTANT)
            video.write(img_)
            print('written',ex)
        if show_pipeline:
            print('step4: img size',img.shape)
            cv2.imshow('window', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    if record:
        cv2.destroyAllWindows()
        video.release()
    return [],[]


    for batch_idx, (x, labels, im_name) in enumerate(tqdm(loader)):
        

        #print(im_name)
        #x_dnn=np.moveaxis(x.numpy().astype(np.uint8),[1,2,3],[3,1,2])
        x_dnn=np.moveaxis(x.numpy(),[1,2,3],[3,1,2])
        #x_dnn*255 = 416 size, resized with border
        
        x = x.to(device)
        #with torch.no_grad():
        #    predictions = model(x)
        #blob = cv2.dnn.blobFromImages(x_dnn, scalefactor=0.00392, size=(416, 416), mean=(0, 0, 0), swapRB=True, crop=False)

        blob = cv2.dnn.blobFromImages(x_dnn, scalefactor=1, size=(416, 416), mean=(0, 0, 0), swapRB=False, crop=False)
        model.setInput(blob)
        outputs = model.forward(outputLayers)        
        
        #model.setInput(x.numpy())
        #outputs=model.forward(outputLayers)


        # output is [a_1,a_2,a_3] with a_i = numpy.ndarray of float32
        # a_1 has size (bsize,507,7)  , 507 = 13*13*3, yolo gridscale 13, 3 anchor boxes per gridcell, output vector dim=7 (p_c,x,y,w,h,c1,c2)
        # a_2 has size (bsize,2028,7) ,2028 = 26*26*3
        # a_3 has size (bsize,8112,7) ,8112 = 52*52*3

        # STILL NEED TO FIX CODE BELOW
        # predictions below is [a_1,a_2,a_3] with a_i = torch tensor
        # a_1 has size (bsize,3,13,13,7) where 3=#anchor boxes per gridcell, 13*13 gridsize, output vector dim=7,
        # 
        #  
        
        batch_size = x.shape[0]
        predictions=[
            torch.tensor(outputs[0].reshape(batch_size,3,13,13,7)).to(device), # TO DO: derive dimensions from data, not hardcoded
            torch.tensor(outputs[1].reshape(batch_size,3,26,26,7)).to(device),
            torch.tensor(outputs[2].reshape(batch_size,3,52,52,7)).to(device),
        ]
        #return [],[]
        bboxes = [[] for _ in range(batch_size)]
        for i in range(3):
            S = predictions[i].shape[2]
            anchor = torch.tensor([*anchors[i]]).to(device) * S
            boxes_scale_i = cells_to_bboxes(
                predictions[i], anchor, S=S, is_preds=True
            )
            for idx, (box) in enumerate(boxes_scale_i):
                bboxes[idx] += box

        # we just want one bbox for each label, not one for each scale
        true_bboxes = cells_to_bboxes(
            labels[2], anchor, S=S, is_preds=False
        )

        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                bboxes[idx],
                iou_threshold=iou_threshold,
                threshold=threshold,
                box_format=box_format,
            )

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1

    #model.train()
    return all_pred_boxes, all_true_boxes

def cells_to_bboxes(predictions, anchors, S, is_preds=True):
    """
    Scales the predictions coming from the model to
    be relative to the entire image such that they for example later
    can be plotted or.
    INPUT:
    predictions: tensor of size (N, 3, S, S, num_classes+5)
    anchors: the anchors used for the predictions
    S: the number of cells the image is divided in on the width (and height)
    is_preds: whether the input is predictions or the true bounding boxes
    OUTPUT:
    converted_bboxes: the converted boxes of sizes (N, num_anchors, S, S, 1+5) with class index,
                      object score, bounding box coordinates
    """
    BATCH_SIZE = predictions.shape[0]
    num_anchors = len(anchors)
    box_predictions = predictions[..., 1:5]
    if is_preds:
        anchors = anchors.reshape(1, len(anchors), 1, 1, 2)
        box_predictions[..., 0:2] = torch.sigmoid(box_predictions[..., 0:2])
        box_predictions[..., 2:] = torch.exp(box_predictions[..., 2:]) * anchors
        scores = torch.sigmoid(predictions[..., 0:1])
        #CHECK!!!!
        #scores=predictions[...,0:1]
        best_class = torch.argmax(predictions[..., 5:], dim=-1).unsqueeze(-1)
    else:
        scores = predictions[..., 0:1]
        best_class = predictions[..., 5:6]

    cell_indices = (
        torch.arange(S)
        .repeat(predictions.shape[0], 3, S, 1)
        .unsqueeze(-1)
        .to(predictions.device)
    )
    x = 1 / S * (box_predictions[..., 0:1] + cell_indices)
    y = 1 / S * (box_predictions[..., 1:2] + cell_indices.permute(0, 1, 3, 2, 4))
    w_h = 1 / S * box_predictions[..., 2:4]
    converted_bboxes = torch.cat((best_class.to(torch.float32), scores, x, y, w_h), dim=-1).reshape(BATCH_SIZE, num_anchors * S * S, 6)
    return converted_bboxes.tolist()

def check_class_accuracy(model, loader, threshold):
    model.eval()
    tot_class_preds, correct_class = 0, 0
    tot_noobj, correct_noobj = 0, 0
    tot_obj, correct_obj = 0, 0

    for idx, (x, y, im_name) in enumerate(tqdm(loader)):
        #if idx == 100:
        #    break
        x = x.to(config.DEVICE)
        with torch.no_grad():
            out = model(x)

        for i in range(3):
            y[i] = y[i].to(config.DEVICE)
            obj = y[i][..., 0] == 1 # in paper this is Iobj_i
            noobj = y[i][..., 0] == 0  # in paper this is Iobj_i

            correct_class += torch.sum(
                torch.argmax(out[i][..., 5:][obj], dim=-1) == y[i][..., 5][obj]
            )
            tot_class_preds += torch.sum(obj)

            obj_preds = torch.sigmoid(out[i][..., 0]) > threshold
            correct_obj += torch.sum(obj_preds[obj] == y[i][..., 0][obj])
            tot_obj += torch.sum(obj)
            correct_noobj += torch.sum(obj_preds[noobj] == y[i][..., 0][noobj])
            tot_noobj += torch.sum(noobj)

    print(f"Class accuracy is: {(correct_class/(tot_class_preds+1e-16))*100:2f}%")
    print(f"No obj accuracy is: {(correct_noobj/(tot_noobj+1e-16))*100:2f}%")
    print(f"Obj accuracy is: {(correct_obj/(tot_obj+1e-16))*100:2f}%")
    model.train()


def get_mean_std(loader):
    # var[X] = E[X**2] - E[X]**2
    channels_sum, channels_sqrd_sum, num_batches = 0, 0, 0

    for data, _ in tqdm(loader):
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_sqrd_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_sqrd_sum / num_batches - mean ** 2) ** 0.5

    return mean, std


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def get_loaders(train_csv_path, test_csv_path):
    from dataset import YOLODataset

    IMAGE_SIZE = config.IMAGE_SIZE
    train_dataset = YOLODataset(
        train_csv_path,
        transform=config.train_transforms,
        S=[IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8],
        img_dir=config.IMG_DIR,
        label_dir=config.LABEL_DIR,
        anchors=config.ANCHORS,
    )
    test_dataset = YOLODataset(
        test_csv_path,
        transform=config.test_transforms,
        S=[IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8],
        img_dir=config.IMG_DIR,
        label_dir=config.LABEL_DIR,
        anchors=config.ANCHORS,
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=False,
        drop_last=False,
    )

    eval_dataset = YOLODataset(
        test_csv_path,
        transform=config.test_transforms,
        S=[IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8],
        img_dir=config.IMG_DIR,
        label_dir=config.LABEL_DIR,
        anchors=config.ANCHORS,
        sort_imgs=True
    )
    eval_loader = DataLoader(
        dataset=eval_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=False,
        drop_last=False,
    )

    return train_loader, test_loader, eval_loader

def plot_couple_examples(model, loader, thresh, iou_thresh, anchors):
    model.eval()
    x, y, _ = next(iter(loader))
    if torch.cuda.is_available():
        x = x.to("cuda")
    with torch.no_grad():
        out = model(x)
        bboxes = [[] for _ in range(x.shape[0])]
        for i in range(3):
            batch_size, A, S, _, _ = out[i].shape
            anchor = anchors[i]
            boxes_scale_i = cells_to_bboxes(
                out[i], anchor, S=S, is_preds=True
            )
            for idx, (box) in enumerate(boxes_scale_i):
                bboxes[idx] += box

        model.train()

    for i in range(x.shape[0]):
        nms_boxes = non_max_suppression(
            bboxes[i], iou_threshold=iou_thresh, threshold=thresh, box_format="midpoint",
        )
        plot_image(x[i].permute(1,2,0).detach().cpu(), nms_boxes)



def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
