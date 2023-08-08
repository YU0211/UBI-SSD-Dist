import torch
from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from Dataset.ubi_dataset import UBI_Dataset
from vision.utils import box_utils, measurements
from vision.utils.misc import str2bool, Timer
from datetime import datetime, timedelta
import argparse
import pathlib
import numpy as np
import logging
import sys
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from tqdm import tqdm
from torchsummary import summary

parser = argparse.ArgumentParser(description="SSD Evaluation on UBI Dataset.")
parser.add_argument('--Skip_infer', action='store_true', help='Do not need to inferance')
parser.add_argument('--net', default="mb2-ssd-lite",
                    help="The network architecture, it should be of mb1-ssd, mb1-ssd-lite, mb2-ssd-lite or vgg16-ssd.")
parser.add_argument("--trained_model", default="models/best.pth", type=str)

parser.add_argument("--dataset_type", default="ubi", type=str,
                    help='Specify dataset type. Currently support ubi.')
parser.add_argument("--dataset", type=str,
                    help="The root directory of the UBI dataset.")
parser.add_argument("--label_file", type=str, help="The label file path.")
parser.add_argument("--use_cuda", type=str2bool, default=True)
parser.add_argument("--use_2007_metric", type=str2bool, default=False)
parser.add_argument("--nms_method", type=str, default="hard")
parser.add_argument("--iou_threshold", type=float, default=0.5,
                    help="The threshold of Intersection over Union.")
parser.add_argument("--eval_dir", default="eval_results",
                    type=str, help="The directory to store evaluation results.")
parser.add_argument('--mb2_width_mult', default=1.0, type=float,
                    help='Width Multiplifier for MobilenetV2')
args = parser.parse_args()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available()
                      and args.use_cuda else "cpu")


def group_annotation_by_class(dataset):
    true_case_stat = {}
    all_gt_boxes = {}
    all_difficult_cases = {}
    for i in tqdm(range(len(dataset))):
        image_id, annotation = dataset.get_annotation(i)
        gt_boxes, classes, is_difficult = annotation
        gt_boxes = torch.from_numpy(gt_boxes)

        for i, difficult in enumerate(is_difficult):
            class_index = int(classes[i])
            gt_box = gt_boxes[i]
            if not difficult:
                true_case_stat[class_index] = true_case_stat.get(
                    class_index, 0) + 1

            if class_index not in all_gt_boxes:
                all_gt_boxes[class_index] = {}
            if image_id not in all_gt_boxes[class_index]:
                all_gt_boxes[class_index][image_id] = []
            all_gt_boxes[class_index][image_id].append(gt_box)

            if class_index not in all_difficult_cases:
                all_difficult_cases[class_index] = {}
            if image_id not in all_difficult_cases[class_index]:
                all_difficult_cases[class_index][image_id] = []
            all_difficult_cases[class_index][image_id].append(difficult)

    for class_index in all_gt_boxes:
        for image_id in all_gt_boxes[class_index]:
            all_gt_boxes[class_index][image_id] = torch.stack(
                all_gt_boxes[class_index][image_id])

    for class_index in all_difficult_cases:
        for image_id in all_difficult_cases[class_index]:
            all_gt_boxes[class_index][image_id] = torch.as_tensor(
                all_gt_boxes[class_index][image_id])

    return true_case_stat, all_gt_boxes, all_difficult_cases


def compute_average_precision_per_class(num_true_cases, gt_boxes, difficult_cases,
                                        prediction_file, iou_threshold, use_2007_metric):
    with open(prediction_file) as f:
        image_ids = []
        boxes = []
        scores = []
        for line in f:
            t = line.rstrip().split(" ")
            image_ids.append(t[0])
            scores.append(float(t[1]))
            box = torch.tensor([float(v) for v in t[2:]]).unsqueeze(0)
            boxes.append(box)
        scores = np.array(scores)
        sorted_indexes = np.argsort(-scores)
        boxes = [boxes[i] for i in sorted_indexes]
        image_ids = [image_ids[i] for i in sorted_indexes]
        true_positive = np.zeros(len(image_ids))
        false_positive = np.zeros(len(image_ids))
        matched = set()

        for i, image_id in enumerate(image_ids):
            box = boxes[i]
            if image_id not in gt_boxes:
                false_positive[i] = 1
                continue

            gt_box = gt_boxes[image_id]
            ious = box_utils.iou_of(box, gt_box)
            max_iou = torch.max(ious).item()
            max_arg = torch.argmax(ious).item()
            if max_iou > iou_threshold:
                if difficult_cases[image_id][max_arg] == 0:
                    if (image_id, max_arg) not in matched:
                        true_positive[i] = 1
                        matched.add((image_id, max_arg))
                    else:
                        false_positive[i] = 1
            else:
                false_positive[i] = 1

    true_positive = true_positive.cumsum()
    false_positive = false_positive.cumsum()
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / num_true_cases

    if use_2007_metric:
        return precision, recall, measurements.compute_voc2007_average_precision(precision, recall)
    else:
        return precision, recall, measurements.compute_average_precision(precision, recall)


if __name__ == '__main__':

    timer = Timer()
    class_names = [name.strip() for name in open(args.label_file).readlines()]

    if args.dataset_type == "ubi":
        dataset = UBI_Dataset(args.dataset, dataset_type='test')
    else:
        raise ValueError(f"Dataset type {args.dataset_type} is not supported.")

    print("---> Processing ground truth data")
    true_case_stat, all_gt_boxes, all_difficult_cases = group_annotation_by_class(
        dataset)
    print("---> Done")
    
    if args.Skip_infer:
        print("\n---> Skip inferance")
        eval_path = pathlib.Path(args.eval_dir)
    else:
        dt = datetime.now() + timedelta(hours=8)
        eval_path = pathlib.Path(f'{args.eval_dir}/{str(dt)}')   
        eval_path.mkdir(parents=True, exist_ok=True)
        
        if args.net == 'vgg16-ssd':
            net = create_vgg_ssd(len(class_names), is_test=True)
        elif args.net == 'mb1-ssd':
            net = create_mobilenetv1_ssd(len(class_names), is_test=True)
        elif args.net == 'mb1-ssd-lite':
            net = create_mobilenetv1_ssd_lite(len(class_names), is_test=True)
        elif args.net == 'mb2-ssd-lite':
            net = create_mobilenetv2_ssd_lite(
                len(class_names), width_mult=args.mb2_width_mult, is_test=True)
        else:
            logging.fatal("The net type is wrong.")
            parser.print_help(sys.stderr)
            sys.exit(1)

        timer.start("Load Model")
        net.load(args.trained_model)
        net = net.to(DEVICE)
        print(f'\nIt took {timer.end("Load Model")} seconds to load the model.\n')
        summary(net, input_size=(3, 300, 300))
        if args.net == 'vgg16-ssd':
            predictor = create_vgg_ssd_predictor(
                net, nms_method=args.nms_method, device=DEVICE)
        elif args.net == 'mb1-ssd':
            predictor = create_mobilenetv1_ssd_predictor(
                net, nms_method=args.nms_method, device=DEVICE)
        elif args.net == 'mb1-ssd-lite':
            predictor = create_mobilenetv1_ssd_lite_predictor(
                net, nms_method=args.nms_method, device=DEVICE)
        elif args.net == 'mb2-ssd-lite' or args.net == "mb3-large-ssd-lite" or args.net == "mb3-small-ssd-lite":
            predictor = create_mobilenetv2_ssd_lite_predictor(
                net, nms_method=args.nms_method, device=DEVICE)
        else:
            logging.fatal(
                "The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
            parser.print_help(sys.stderr)
            sys.exit(1)

        results = []
        print("\n---> Start inference")
        timer.start("Inference")
        for i in tqdm(range(len(dataset))):
            image = dataset.get_image(i)
            boxes, labels, probs = predictor.predict(image)
            indexes = torch.ones(labels.size(0), 1, dtype=torch.float32) * i
            if boxes.nelement() == 0:
                continue
            else:
                results.append(torch.cat([
                    indexes.reshape(-1, 1),
                    labels.reshape(-1, 1).float(),
                    probs.reshape(-1, 1),
                    boxes
                ], dim=1))
        mean_infer_time = timer.end("Inference") / len(dataset)
        print("---> Done")
        
        results = torch.cat(results)
        print("\n---> Save inference result")
        for class_index, class_name in enumerate(class_names):
            if class_index == 0:
                continue  # ignore background
            prediction_path = eval_path / f"det_test_{class_name}.txt"
            with open(prediction_path, "w") as f:
                sub = results[results[:, 1] == class_index, :]
                for i in range(sub.size(0)):
                    prob_box = sub[i, 2:].numpy()
                    image_id = dataset.ids[int(sub[i, 0])]
                    print(
                        image_id + " " + " ".join([str(v) for v in prob_box]),
                        file=f
                    )
    aps = []
    print("\n---> Calculate Metric")
    print(f"{'Evaluate result on IOU_50:':<30} {'Precision':<10} {'Recall':<10} {'AP':<8}")
    for class_index in true_case_stat.keys():
        if class_index == 0:
            continue

        prediction_path = eval_path / \
            f"det_test_{class_names[class_index]}.txt"
        precision, recall, ap = compute_average_precision_per_class(
            true_case_stat[class_index],
            all_gt_boxes[class_index],
            all_difficult_cases[class_index],
            prediction_path,
            args.iou_threshold,
            args.use_2007_metric
        )
        print(
            f"{class_names[class_index]:<30} {precision[-1]:<10.4f} {recall[-1]:<10.4f} {ap:<8.4f}")
        aps.append(ap)

    print(
        f"\nAverage Precision Across All Classes(mAP): {sum(aps)/len(aps):.4f}")
    if not(args.Skip_infer):
        print("Inference per image: {:.4f} seconds.".format(mean_infer_time))
    print()
