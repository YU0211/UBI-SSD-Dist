import numpy as np
import os, sys
import re
import math
import random
import cv2
from tqdm import tqdm
import time
from multiprocessing import Pool
from rknn.api import RKNN

INPUT_SIZE = 300
ORIG_IMG_SIZE = (1920, 1080)

NUM_RESULTS = 1917
NUM_CLASSES = 91

Y_SCALE = 10.0
X_SCALE = 10.0
H_SCALE = 5.0
W_SCALE = 5.0

RKNN_MODEL = './ssd_mobilenet_v1_coco.rknn'


def expit(x):
    return 1. / (1. + math.exp(-x))


def unexpit(y):
    return -1.0 * math.log((1.0 / y) - 1.0)


def CalculateOverlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1):
    w = max(0.0, min(xmax0, xmax1) - max(xmin0, xmin1))
    h = max(0.0, min(ymax0, ymax1) - max(ymin0, ymin1))
    i = w * h
    u = (xmax0 - xmin0) * (ymax0 - ymin0) + \
        (xmax1 - xmin1) * (ymax1 - ymin1) - i

    if u <= 0.0:
        return 0.0

    return i / u


def load_box_priors():
    box_priors_ = []
    fp = open('./box_priors.txt', 'r')
    ls = fp.readlines()
    for s in ls:
        aList = re.findall('([-+]?\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?', s)
        for ss in aList:
            aNum = float((ss[0]+ss[2]))
            box_priors_.append(aNum)
    fp.close()

    box_priors = np.array(box_priors_)
    box_priors = box_priors.reshape(4, NUM_RESULTS)

    return box_priors


if __name__ == '__main__':

    # Create RKNN object
    rknn = RKNN()

    # Config for Model Input PreProcess
    print('--> Config model')
    rknn.config(target_platform='rv1126', mean_values=[[127.5, 127.5, 127.5]], std_values=[
                [127.5, 127.5, 127.5]], reorder_channel='0 1 2')
    print('done')

    # Direct Load RKNN Model
    print('--> Load model')
    ret = rknn.load_rknn(RKNN_MODEL)
    if ret != 0:
        print('Load RKNN model failed!')
        exit(ret)
    print('done')

    # Set inputs
    print('--> Load file')
    all_class_name = []
    with open(f'coco_labels_list.txt', 'r') as f:
        for line in f.readlines():
            line = line.rstrip()
            if line in ['bus', 'car', 'truck']:
                all_class_name.append('vehicle')
            elif line == 'motorcycle':
                all_class_name.append('rider')
            elif line == 'person':
                all_class_name.append('pedestrian')
            else:
                all_class_name.append(line)

    with open(f'../../Data/test/test.txt', 'r') as f:
        all_image_name = [line.rstrip() for line in f.readlines()]

    inputs = []
    if os.path.exists('test_imgs.npy') == False:
        for i in tqdm(all_image_name):
            path = os.path.join('../../Data/test/images', i + '.jpg')
            orig_img = cv2.imread(path)
            img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE),
                             interpolation=cv2.INTER_CUBIC)
            inputs.append(img)
        np.save('test_imgs.npy', inputs)
    else:
        inputs = np.load('test_imgs.npy')
    print('done')
    print(f'Totol images:{len(inputs)}')
    
    # init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime(async_mode=True)
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    print('done')

    # Inference
    print('--> Running model')
    outputs = []
    
    def rknn_infer(img):
        rknn.init_runtime(async_mode=True)
        return rknn.inference(inputs=[img])
    
    with Pool(processes=8) as p:
        for output in tqdm(p.imap(rknn_infer, inputs), total=len(inputs)):
            outputs.append(output)

#     for img in tqdm(inputs):
#          outputs.append(rknn.inference(inputs=[img]))
    
    print(len(outputs))
    for img_id, output in tqdm(enumerate(outputs), total=len(outputs)):
        predictions = output[0].reshape((1, NUM_RESULTS, 4))
        outputClasses = output[1].reshape(
            (1, NUM_RESULTS, NUM_CLASSES))
        candidateBox = np.zeros([2, NUM_RESULTS], dtype=int)
        candidateProbs = np.zeros([1, NUM_RESULTS], dtype=float)
        vaildCnt = 0

        box_priors = load_box_priors()

        # Post Process
        # got valid candidate box
        for i in range(0, NUM_RESULTS):
            topClassScore = -1000
            topClassScoreIndex = -1

            # Skip the first catch-all class.
            for j in range(1, NUM_CLASSES):
                score = expit(outputClasses[0][i][j])

                if score > topClassScore:
                    topClassScoreIndex = j
                    topClassScore = score

            if topClassScore > 0.4:
                candidateBox[0][vaildCnt] = i  # index of box
                # index of class
                candidateBox[1][vaildCnt] = topClassScoreIndex
                candidateProbs[0][vaildCnt] = topClassScore  # probs
                vaildCnt += 1

        # calc position
        for i in range(0, vaildCnt):

            n = candidateBox[0][i]
            ycenter = predictions[0][n][0] / Y_SCALE * \
                box_priors[2][n] + box_priors[0][n]
            xcenter = predictions[0][n][1] / X_SCALE * \
                box_priors[3][n] + box_priors[1][n]
            h = math.exp(predictions[0][n][2] / H_SCALE) * box_priors[2][n]
            w = math.exp(predictions[0][n][3] / W_SCALE) * box_priors[3][n]

            ymin = ycenter - h / 2.
            xmin = xcenter - w / 2.
            ymax = ycenter + h / 2.
            xmax = xcenter + w / 2.

            predictions[0][n][0] = ymin
            predictions[0][n][1] = xmin
            predictions[0][n][2] = ymax
            predictions[0][n][3] = xmax

        # NMS
        for i in range(0, vaildCnt):

            n = candidateBox[0][i]
            xmin0 = predictions[0][n][1]
            ymin0 = predictions[0][n][0]
            xmax0 = predictions[0][n][3]
            ymax0 = predictions[0][n][2]

            for j in range(i+1, vaildCnt):
                m = candidateBox[0][j]

                if m == -1:
                    continue

                xmin1 = predictions[0][m][1]
                ymin1 = predictions[0][m][0]
                xmax1 = predictions[0][m][3]
                ymax1 = predictions[0][m][2]

                iou = CalculateOverlap(xmin0, ymin0, xmax0,
                                       ymax0, xmin1, ymin1, xmax1, ymax1)

                if iou >= 0.45:
                    candidateBox[0][j] = -1

        # Save result
        # candidateBox[] 0: index of box, 1: index of class
        for i in range(0, vaildCnt):
            cls = all_class_name[candidateBox[1][i]]
            COND_1 = candidateBox[0][i] == -1
            COND_2 = cls not in ['vehicle', 'rider', 'pedestrian']

            if COND_1 or COND_2:
                continue

            n = candidateBox[0][i]

            xmin = max(0.0, min(1.0, predictions[0][n][1])) * ORIG_IMG_SIZE[0]
            ymin = max(0.0, min(1.0, predictions[0][n][0])) * ORIG_IMG_SIZE[1]
            xmax = max(0.0, min(1.0, predictions[0][n][3])) * ORIG_IMG_SIZE[0]
            ymax = max(0.0, min(1.0, predictions[0][n][2])) * ORIG_IMG_SIZE[1]
            result = [all_image_name[img_id],
                      candidateProbs[0][i], xmin, ymin, xmax, ymax]
            with open(f"../outputs/det_test_{cls}.txt", "a") as f:
                f.write(" ".join([str(r) for r in result]) + '\n')

    # Evaluate Perf on Simulator
    print('--> Evaluate model performance')
    rknn.init_runtime()
    rknn.eval_perf(inputs=[inputs[0]], is_print=True)
    print('done')

    # Release RKNN Context
    rknn.release()
