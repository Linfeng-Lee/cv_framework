import os
import time
import shutil
import argparse
import numpy as np
import cv2
from tqdm import tqdm
import torch
from PIL import Image
from loguru import logger
from easydict import EasyDict as edict
from torchvision import transforms
import network
from utils.util import write_json, get_images, path_join, load_yaml


def export_torchscript(args, weight: str, save_path: str):
    logger.info("export torchscript...")
    logger.info(f"load weights:{weight}")

    rand_input = torch.rand(1, 3, args.input_h, args.input_w).cuda()

    from export.shufflenetv2_embedding import shufflenet_v2_x1_0

    # model = network.__dict__[args.net](pretrained=False, embedding_classes=args.classes)

    model = shufflenet_v2_x1_0(embedding_classes=args.classes)
    checkpoint = torch.load(weight)
    static_dict = checkpoint['state_dict']
    model.load_state_dict(static_dict, strict=True)

    model.cuda()
    model.eval()

    torchscript = torch.jit.trace(model, rand_input, strict=False)
    localtime = time.localtime(time.time())
    date = "-".join([str(i) for i in localtime[0:3]])

    file_name = "{}_{}_{}x{}_{}.torchscript.pt".format(args.net,
                                                       str(args.classes),
                                                       str(args.input_w),
                                                       str(args.input_h),
                                                       date)
    save_path = path_join(save_path, file_name)
    torchscript.save(save_path)
    logger.success(f"save_path:{save_path}")
    logger.success("ok")


def export_onnx(args, weight: str, save_path: str):
    logger.info("export onnx...")
    rand_input = torch.rand(1, 3, args.input_h, args.input_w).cpu()

    from export.shufflenetv2_segmantation import shufflenet_v2_x1_0

    model = shufflenet_v2_x1_0(num_classes=args.classes)
    # model_path = args.data.replace("params", "models/checkpoint.pth.tar")
    checkpoint = torch.load(weight)
    static_dict = checkpoint['state_dict']
    model.load_state_dict(static_dict, strict=True)
    model.cpu()
    model.eval()

    localtime = time.localtime(time.time())
    date = "-".join([str(i) for i in localtime[0:3]])

    file_name = "{}_{}_{}x{}_{}.onnx".format(args.net,
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


class ScriptModel():
    def __init__(self, model_path, cls_, h, w):
        self.model = torch.jit.load(model_path)
        # checkpoint = torch.load(model_path) 
        # static_dict = checkpoint['state_dict']
        # self.models.load_state_dict(static_dict, strict=False)
        self.h = h
        self.w = w

    def inference(self, path):
        self.model.cuda()
        self.model.eval()
        inputs = self._preprocess(path)
        outputs = self.model(inputs)

        return outputs

    def _preprocess(self, image):
        # preprocess
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        normalize = transforms.Normalize(mean=mean,
                                         std=std)
        transforms_ = transforms.Compose([
            transforms.Resize((self.h, self.w)),
            transforms.ToTensor(),
            normalize,
        ])
        img = cv2.imread(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_color = Image.fromarray(img)
        inputs = transforms_(img_color)
        inputs = inputs.unsqueeze(0)
        inputs = inputs.cuda()

        return inputs

    def test_inference_time(self, batchsize, repeat_time=10):
        self.model.cuda()
        self.model.eval()
        for _ in range(repeat_time):
            inputs = torch.rand(batchsize, 3, self.h, self.w).cuda()
            start = time.time()
            outputs = self.model(inputs)
            end = time.time()
            print("batchsize " + str(batchsize) + " inference time:", (end - start) * 1000, " ms")

        return outputs


def export_template_json(config, weight: str, save_path: str):
    cls_ = config.classes
    w, h = config.input_w, config.input_h
    # model_path = r"export/embedding_shufflenet_v2_x1_0_11_320x192_2022-10-27.torchscript.pt"
    # input_dir = r"data/train"

    model_path = weight
    input_dir = path_join(config.data_root, 'train')

    types = os.listdir(input_dir)
    model = ScriptModel(model_path, cls_, h, w)

    emb_vecs = {}

    for t in types:
        i = int(t.split("_")[0])
        if i == 0:
            continue
        images = get_images(os.path.join(input_dir, t))
        vecs = []
        for image in tqdm(images):
            _ = cv2.imread(image)
            if _ is None:
                shutil.copy(image, "temp")
                print("bad image")
                continue
            outputs, embedding = model.inference(image)
            # print(embedding.norm())
            vecs.append(embedding.detach().cpu())
            # print(emb_vec.shape)
            # print(emb_vec.sum())
            # exit()
            # print(emb_vec[0].detach().cpu().numpy().tolist())
            # emb_vecs[image]=emb_vec[0].detach().cpu().numpy().tolist()
            # emb_vecs[i]=emb_vec[0].detach().cpu().numpy().tolist()

        if vecs:
            vec = torch.cat(vecs, dim=0)
            print(vec.shape)
            vec = vec.mean(dim=0)
            print(vec.shape)
            emb_vecs[i] = vec.numpy().tolist()
            print(len(emb_vecs[i]))

    localtime = time.localtime(time.time())
    date = "-".join([str(i) for i in localtime[0:3]])

    save_path = path_join(save_path, f'template_norm_{date}.json')
    write_json(emb_vecs, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training yaml config.')
    parser.add_argument('--yaml', type=str, default='params/shuangjing.yaml',
                        help='input your training yaml file.')
    parser.add_argument('--weight', type=str, default='project/shuangjing/model/checkpoint.pth.tar',
                        help='input your weight.')
    parser.add_argument('--type', type=str, default='torchscript',
                        help='e.g.torchscript,onnx,embedding_cls')
    args = parser.parse_args()

    config = load_yaml(args.yaml)
    config = edict(config)

    export_path = os.path.join('project', config.project, 'export')

    if args.type == 'torchscript':
        export_path = os.path.join('project', config.project, 'export')
        export_torchscript(config, args.weight, export_path)

    if args.type == 'segmentation' and args.type == 'onnx':
        export_onnx(config, args.weight, export_path)

    if args.type == 'embedding_cls':
        export_template_json(config, args.weight, export_path)
