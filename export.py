import sys
import argparse
import datetime
import time
from pathlib import Path

import torch
from torch.utils.mobile_optimizer import optimize_for_mobile

from vision.utils.logger import init_logger
from vision.ssd.vgg_ssd import create_vgg_ssd
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite




def compute_file_size(file):
    return Path(file).stat().st_size / 1e6


def export_torchscript(model, img, file, optimize):
    # TorchScript model export
    prefix = 'TorchScript:'
    try:
        print(f'\n{prefix} starting export with torch {torch.__version__}...')
        f = str(file.with_suffix(
            '.torchscript_optim.pt' if optimize else '.torchscript.pt'))
        ts = torch.jit.trace(model, img, strict=False)
        (optimize_for_mobile(ts) if optimize else ts).save(f)
        print(f'{prefix} export success, saved as {f} ({compute_file_size(f):.1f} MB)')
        return ts
    except Exception as e:
        print(f'{prefix} export failure: {e}')


# def export_onnx(model, img, file, opset, train, dynamic, simplify):
#     # ONNX model export
#     prefix = colorstr('ONNX:')
#     try:
#         check_requirements(('onnx',))
#         import onnx

#         print(f'\n{prefix} starting export with onnx {onnx.__version__}...')
#         f = file.with_suffix('.onnx')
#         torch.onnx.export(model, img, f, verbose=False, opset_version=opset,
#                           training=torch.onnx.TrainingMode.TRAINING if train else torch.onnx.TrainingMode.EVAL,
#                           do_constant_folding=not train,
#                           input_names=['images'],
#                           output_names=['output'],
#                           dynamic_axes={'images': {0: 'batch', 2: 'height', 3: 'width'},  # shape(1,3,640,640)
#                                         'output': {0: 'batch', 1: 'anchors'}  # shape(1,25200,85)
#                                         } if dynamic else None)

#         # Checks
#         model_onnx = onnx.load(f)  # load onnx model
#         onnx.checker.check_model(model_onnx)  # check onnx model
#         # print(onnx.helper.printable_graph(model_onnx.graph))  # print

#         # Simplify
#         if simplify:
#             try:
#                 check_requirements(('onnx-simplifier',))
#                 import onnxsim

#                 print(f'{prefix} simplifying with onnx-simplifier {onnxsim.__version__}...')
#                 model_onnx, check = onnxsim.simplify(
#                     model_onnx,
#                     dynamic_input_shape=dynamic,
#                     input_shapes={'images': list(img.shape)} if dynamic else None)
#                 assert check, 'assert check failed'
#                 onnx.save(model_onnx, f)
#             except Exception as e:
#                 print(f'{prefix} simplifier failure: {e}')
#         print(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
#         print(f"{prefix} run --dynamic ONNX model inference with: 'python detect.py --weights {f}'")
#     except Exception as e:
#         print(f'{prefix} export failure: {e}')


def run(
        net_type='mb2-ssd-lite',
        weight='./models/best.pth',  # weight path
        img_size=(300, 300),  # image (height, width)
        batch_size=1,  # batch size
        include=('torchscript', 'onnx'),  # include formats
        half=True,  # FP16 half-precision export
        train=False,  # model.train() mode
        optimize=False,  # TorchScript: optimize for mobile
        # dynamic=False,  # ONNX: dynamic axes
        # simplify=False,  # ONNX: simplify model
        # opset=12,  # ONNX: opset version
):
    t = time.time()
    include = [x.lower() for x in include]
    img_size *= 2 if len(img_size) == 1 else 1  # expand
    file = Path(weight)

    # Load PyTorch model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    logger.info(f"Device on {device}")

    assert not (
        device.type == 'cpu' and half), '--half only compatible with GPU export, i.e. use --device 0'

    label_path = "models/ubi-model-labels.txt"

    class_names = [name.strip() for name in open(label_path).readlines()]
    print(len(class_names))
    if net_type == 'vgg16-ssd':
        model = create_vgg_ssd(len(class_names), is_test=True)
    elif net_type == 'mb1-ssd':
        model = create_mobilenetv1_ssd(len(class_names), is_test=True)
    elif net_type == 'mb2-ssd-lite':
        model = create_mobilenetv2_ssd_lite(len(class_names), is_test=True)
    else:
        logger.fatal(
            "The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
        sys.exit(1)

    logger.info(
        f"PyTorch: loading model {net_type} from {weight} ({compute_file_size(weight):.1f} MB)")
    model.load(weight)
    model.to(device)
    # Input
    # image size(1,3,320,192) iDetection
    img = torch.zeros(batch_size, 3, *img_size).to(device)

    # Update model
    if half:
        img, model = img.half(), model.half()  # to FP16
    # training mode = no Detect() layer grid construction
    model.train() if train else model.eval()

    # for _ in range(2):
    #     y = model(img)  # dry runs

    # Exports
    if 'torchscript' in include:
        export_torchscript(model, img, file, optimize)
    # if 'onnx' in include:
    #     export_onnx(model, img, file, opset, train, dynamic, simplify)

    # Finish
    print(f'\nExport complete ({time.time() - t:.2f}s)'
          f"\nResults saved to { file.parent.resolve()}"
          f'\nVisualize with https://netron.app')


def parse_opt():
    parser = argparse.ArgumentParser(description='Export SSD model')
    parser.add_argument('--net_type', default="mb2-ssd-lite",
                        help="The network architecture, it can be mb1-ssd, mb2-ssd-lite or vgg16-ssd.")
    parser.add_argument('--weight', type=str,
                        default='./models/best.pth', help='weight path')
    # parser.add_argument('--img_size', nargs='+', type=int, default=[300, 300], help='image (height, width)') only 300*300
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--include', nargs='+',
                        default=['torchscript', 'onnx', 'coreml'], help='include formats')
    parser.add_argument('--half', action='store_true',
                        help='FP16 half-precision export')
    parser.add_argument('--train', action='store_true',
                        help='model.train() mode')
    parser.add_argument('--optimize', action='store_true',
                        help='TorchScript: optimize for mobile')
    # parser.add_argument('--dynamic', action='store_true', help='ONNX: dynamic axes')
    # parser.add_argument('--simplify', action='store_true', help='ONNX: simplify model')
    # parser.add_argument('--opset', type=int, default=13, help='ONNX: opset version')
    return parser.parse_args()


def GMT_8(sec, what):
    GMT_8_time = datetime.datetime.now() + datetime.timedelta(hours=8)
    return GMT_8_time.timetuple()


if __name__ == "__main__":
    opt = parse_opt()

    logger = init_logger()
    logger.info(
        'export: ' + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))

    run(**vars(opt))
