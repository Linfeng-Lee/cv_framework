import os
import time
import argparse

from easydict import EasyDict as edict
from loguru import logger
from utils.common import load_yaml
import torch


def export_torchscript(args, save_path):
    logger.info("export torchscript...")
    rand_input = torch.rand(1, 3, args.input_h, args.input_w).cuda()

    from export.shufflenetv2 import shufflenet_v2_x1_0

    model = shufflenet_v2_x1_0(num_classes=args.classes)
    # model_path = args.data.replace("params", "models/checkpoint.pth.tar")
    checkpoint = torch.load(args.resume)
    static_dict = checkpoint['state_dict']
    model.load_state_dict(static_dict, strict=True)

    model.cuda()
    model.eval()

    torchscript = torch.jit.trace(model, rand_input, strict=False)
    localtime = time.localtime(time.time())
    date = "-".join([str(i) for i in localtime[0:3]])

    file_name = "{}_{}_{}x{}_{}.torchscript.pt".format(args.model_names,
                                                       str(args.classes),
                                                       str(args.input_w),
                                                       str(args.input_h),
                                                       date)
    torchscript.save(os.path.join(save_path, file_name))
    logger.success("ok")


def export_onnx(args, save_path):
    logger.info("export onnx...")
    rand_input = torch.rand(1, 3, args.input_h, args.input_w).cpu()

    from export.shufflenetv2_segmantation import shufflenet_v2_x1_0

    model = shufflenet_v2_x1_0(num_classes=args.classes)
    # model_path = args.data.replace("params", "models/checkpoint.pth.tar")
    checkpoint = torch.load(args.resume)
    static_dict = checkpoint['state_dict']
    model.load_state_dict(static_dict, strict=True)
    model.cpu()
    model.eval()

    localtime = time.localtime(time.time())
    date = "-".join([str(i) for i in localtime[0:3]])

    file_name = "{}_{}_{}x{}_{}.onnx".format(args.model_names,
                                             str(args.classes),
                                             str(args.input_w),
                                             str(args.input_h),
                                             date)
    torch.onnx.export(model,
                      rand_input,
                      os.path.join(save_path, file_name),
                      verbose=False,
                      opset_version=12,
                      input_names=['images'],
                      output_names=['output'])

    logger.success("ok")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training yaml config.')
    parser.add_argument('--yaml', type=str, default='params/shuangjing.yaml', help='input your training yaml file.')
    args = parser.parse_args()

    config = load_yaml(args.yaml)
    args = edict(config)

    export_path = os.path.join('project', args.project, 'export')
    export_torchscript(args, export_path)

    if args.task_type == 'segmentation':
        export_onnx(args, export_path)
