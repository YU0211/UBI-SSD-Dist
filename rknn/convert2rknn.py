import numpy as np
import os
import time
import cv2
from pathlib import Path

from rknn.api import RKNN


PT_MODEL = '/root/ubi/UBI_SSD/models/best.torchscript.pt'
RKNN_MODEL = 'MobilenetV2_SSD_Lite.rknn'
IMG_PATH = '/root/ubi/UBI_SSD/rknn/car mirror_2.jpg'
DATASET = 'dataset.txt'

ORIG_IMG_SIZE = (300, 300)
INPUT_IMG_SIZE = (300, 300)
CLASSES = ['BACKGROUND', 'vehicle', 'rider', 'pedestrian']
QUANTIZE_ON = True


PROB_THRESH = 0.2
IOU_THRESH = 0.45
TOP_K = 10
CANDIDATE_SIZE = 200


def CalculateOverlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1):
    w = max(0.0, min(xmax0, xmax1) - max(xmin0, xmin1))
    h = max(0.0, min(ymax0, ymax1) - max(ymin0, ymin1))
    i = w * h
    u = (xmax0 - xmin0) * (ymax0 - ymin0) + \
        (xmax1 - xmin1) * (ymax1 - ymin1) - i

    if u <= 0.0:
        return 0.0

    return i / u


def area_of(left_top, right_bottom):
    overlap_wh = np.clip(right_bottom - left_top, 0.0, None)
    return overlap_wh[:, 0] * overlap_wh[:, 1]


def cal_iou(bbox, gt, eps=1e-5):
    left_top = np.maximum(bbox[:, :2], gt[:, :2])
    right_bottom = np.minimum(bbox[:, 2:4], gt[:, 2:4])
    overlap_area = area_of(left_top, right_bottom)

    bbox_area = area_of(bbox[:, :2], bbox[:, 2:])
    gt_area = area_of(gt[:, :2], gt[:, 2:])
    return overlap_area / (bbox_area + gt_area - overlap_area + eps)


def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    indexes = np.argsort(-scores)[:candidate_size]
    while len(indexes) > 0:
        current = indexes[0]
        picked.append(current.item())
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[1:]
        rest_boxes = boxes[indexes, :]
        iou = cal_iou(
            rest_boxes,
            np.expand_dims(current_box, axis=0),
        )
        indexes = indexes[iou <= iou_threshold]

    return box_scores[picked, :]


def postprocessing(boxes, scores):
    picked_box_probs = []
    picked_labels = []
    for class_index in range(1, scores.shape[1]):
        probs = scores[:, class_index]
        mask = probs > PROB_THRESH
        probs = probs[mask]
        if probs.shape[0] == 0:
            continue
        subset_boxes = boxes[mask, :]
        box_probs = np.concatenate(
            [subset_boxes, probs.reshape(-1, 1)], axis=1)
        box_probs = hard_nms(box_probs,
                             iou_threshold=IOU_THRESH,
                             top_k=TOP_K,
                             candidate_size=CANDIDATE_SIZE)
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])
    if not picked_box_probs:
        raise (f"Didn't catch the specified object: ${CLASSES}")
    picked_box_probs = np.asarray(picked_box_probs)
    picked_box_probs = np.concatenate(picked_box_probs, axis=0)
    picked_box_probs[:, 0] *= ORIG_IMG_SIZE[0]
    picked_box_probs[:, 1] *= ORIG_IMG_SIZE[1]
    picked_box_probs[:, 2] *= ORIG_IMG_SIZE[0]
    picked_box_probs[:, 3] *= ORIG_IMG_SIZE[1]
    return picked_box_probs[:, :4], picked_labels, picked_box_probs[:, 4]


if __name__ == '__main__':

    # Create RKNN object
    rknn = RKNN()
    # Config for Model Input PreProcess
    print('--> Config model')
    target = 'rv1126'
    device_id = None
    rknn.config(mean_values=[[127.5, 127.5, 127.5]],
                std_values=[[127.5, 127.5, 127.5]],
                reorder_channel='0 1 2',
                target_platform=[target])
    print('done')

    if not Path(RKNN_MODEL).exists():
        # Load PyTorch Model
        print('--> Loading model')
        ret = rknn.load_pytorch(model=PT_MODEL, input_size_list=[
                                [3, INPUT_IMG_SIZE[1], INPUT_IMG_SIZE[0]]])
        if ret != 0:
            print('Load model failed!')
            exit(ret)
        print('done')

        # Build Model
        print('--> Building model')
        ret = rknn.build(do_quantization=QUANTIZE_ON, dataset=DATASET)
        if ret != 0:
            print('Build model failed!')
            exit(ret)
        print('done')

        # Export RKNN Model
        print('--> Export RKNN model')
        rknn.export_rknn(RKNN_MODEL)
        if ret != 0:
            print('Export RKNN model failed!')
            exit(ret)
        print('done')

    # Direct Load RKNN Model
    print('--> Load rknn model')
    ret = rknn.load_rknn(RKNN_MODEL)
    if ret != 0:
        print('Load RKNN model failed!')
        exit(ret)
    print('done')

    # Set inputs
    orig_img = cv2.imread(IMG_PATH)
    orig_img = cv2.resize(orig_img, INPUT_IMG_SIZE,
                          interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

    # init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    print('done')

    # Inference
    print('--> Running model')
    outputs = rknn.inference(inputs=[img])
    print('done')

    scores = outputs[0].squeeze(axis=0)
    boxes = outputs[1].squeeze(axis=0)
    boxes, labels, probs = postprocessing(boxes, scores)

    # Draw result
    color = np.random.uniform(0, 255, size=(3, 3))
    for i in range(boxes.shape[0]):
        box = boxes[i, :]
        label = f"{CLASSES[labels[i]]}: {probs[i]:.2f}"

        i_color = int(labels[i])
        box = [round(b.item()) for b in box]

        cv2.rectangle(orig_img, (box[0], box[1]),
                      (box[2], box[3]), color[i_color], 2)

        cv2.putText(orig_img, label,
                    (box[0] - 10, box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,  # font scale
                    color[i_color],
                    2)  # line type

    cv2.imwrite("out.jpg", orig_img)

    # Evaluate Perf on Simulator
    print('--> Evaluate model performance')
    rknn.eval_perf(inputs=[img], is_print=True)
    print('done')

    # Release RKNN Context
    rknn.release()
