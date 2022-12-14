import shutil
import sys
import torch
import time
from torchvision import transforms
import cv2
from PIL import Image
import numpy as np
import json
import os
import io
import base64
from base64 import b64encode
from tqdm import tqdm
import torch.nn.functional as F
import io


def write_json(json_data, save_path):
    with open(save_path, "w") as f:
        json.dump(json_data, f)


def load_json(json_path):
    with open(json_path) as f:
        json_data = json.load(f)
    return json_data


def img_tobyte(img_pil):
    ENCODING = 'utf-8'
    img_byte = io.BytesIO()
    img_pil.save(img_byte, format='PNG')
    binary_str2 = img_byte.getvalue()
    imageData = base64.b64encode(binary_str2)
    base64_string = imageData.decode(ENCODING)
    return base64_string


class ScriptModel():
    def __init__(self, model_path, cls_, h, w):
        # self.models = torch.jit.load(model_path)

        self.model = torch.load(model_path)

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


def get_images(path):
    ret = []
    for root, dirs, files in os.walk(path):
        for file in files:
            file_extend = file.split(".")[-1]
            # file_head = file.replace("."+file_extend,"")
            if file_extend in ["png"]:
                ret.append(os.path.join(root, file))
    return ret


def get_labelme_ann(image, seg, labelmap, cls_):
    json_data = {}
    shapes = []
    version = "4.5.9"
    flags = {}
    imagePath = image.split(os.sep)[-1]
    img_arr = cv2.imread(image)[:, :, ::-1]
    imageData = img_tobyte(Image.fromarray(img_arr))
    h, w = img_arr.shape[0], img_arr.shape[1]

    for i in range(1, cls_):
        sub_mask = np.zeros((h, w), dtype="uint8")
        sub_mask[seg == i] = 1
        contour, _ = cv2.findContours(sub_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contour:
            if len(cnt) < 3:
                continue
            seg_info = {}
            seg_info["label"] = str(i) + labelmap[i]
            seg_info["points"] = cnt.reshape(-1, 2).tolist()
            # print(seg_info["points"].shape)
            seg_info["group_id"] = None
            seg_info["shape_type"] = "polygon"
            seg_info["flags"] = {}
            shapes.append(seg_info)

    json_data["shapes"] = shapes
    json_data["version"] = version
    json_data["flags"] = flags
    json_data["imagePath"] = imagePath
    json_data["imageData"] = imageData
    json_data["imageHeight"] = h
    json_data["imageWidth"] = w

    return json_data


def cosine_similarity(embeddings1, embeddings2):
    # Distance based on cosine similarity
    embeddings1 = embeddings1[None, :]
    embeddings2 = embeddings2[None, :]

    dot = np.sum(np.multiply(embeddings1, embeddings2), axis=1)
    norm = np.linalg.norm(embeddings1, axis=1) * np.linalg.norm(embeddings2, axis=1)
    similarity = dot / norm
    # dist = np.arccos(similarity) / math.pi

    return similarity


def match_template_vec(embedding, template_vecs):
    score = float("-inf")
    ret = 0
    for k, v in template_vecs.items():
        similarity = cosine_similarity(embedding, np.array(v))
        if similarity > score:
            ret = k
            score = similarity

    return ret


if __name__ == "__main__":
    cls_ = 2
    w, h = 320, 192
    input_dir = r"project/shuangjing/data/test"

    # ??????mode
    model_path = r"E:\??????????????????\??????????????????\??????\???????????????\model_20221018/shufflenetv2_embedding_11_320x192_2022-10-18.torchscript.pt"
    template_path = r'E:\??????????????????\??????????????????\??????\???????????????\model_20221018/template_norm_2022-10-18.json'

    # new models
    # template_path = r"project/shuangjing/export/template_norm_2022-10-27.json"
    # model_path = r"project/shuangjing/export/embedding_shufflenet_v2_x1_0_11_320x192_2022-10-27.torchscript.pt"

    template_vecs = load_json(template_path)

    types = os.listdir(input_dir)
    model = ScriptModel(model_path, cls_, h, w)
    images = get_images(input_dir)
    # emb_vecs={}

    # for t in types:
    #     i=int(t.split("_")[0])
    #     if i==0:
    #         continue
    #     images=get_images(os.path.join(input_dir,t))
    #     vecs=[]
    cnt = 0
    for image in tqdm(images):
        # if cnt > 10:
        #     exit()
        cnt += 1
        _ = cv2.imread(image)

        if _ is None:
            shutil.copy(image, "temp")
            print("bad image")
            continue

        outputs, embedding = model.inference(image)
        gs_score = outputs.softmax(dim=1)[0][0].detach().cpu().item()
        # print(image, gs_score)
        emb = embedding[0].detach().cpu().numpy()

        if gs_score > 0.2:
            defect_idx = match_template_vec(emb, template_vecs)
            print(defect_idx)

            shutil.copy(image, "temp/old_model_all_0.2")
            print(f"bad image:{image} score:{gs_score}")

            # vecs.append(embedding.detach().cpu())
            # print(emb_vec.shape)
            # print(emb_vec.sum())
            # exit()
            # print(emb_vec[0].detach().cpu().numpy().tolist())
            # emb_vecs[image]=emb_vec[0].detach().cpu().numpy().tolist()
            # emb_vecs[i]=emb_vec[0].detach().cpu().numpy().tolist()

    #     if vecs:
    #         vec=torch.cat(vecs,dim=0)
    #         print(vec.shape)
    #         vec=vec.mean(dim=0)
    #         print(vec.shape)
    #         emb_vecs[i]=vec.numpy().tolist()
    #         print(len(emb_vecs[i]))

    # localtime = time.localtime(time.time())
    # date = "-".join([str(i) for i in localtime[0:3]])
    # write_json(emb_vecs,"template_norm_{}.json".format(date))
