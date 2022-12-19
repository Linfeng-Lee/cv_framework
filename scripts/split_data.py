import os
import shutil
from tqdm import tqdm

img_exts = ['.png', '.jpg']


def mode_data(data_root, flag):
    image_files = []
    for sub_root, _, files in os.walk(data_root):
        image_files += [os.path.join(sub_root, f) for f in files if os.path.splitext(f)[-1].lower() in img_exts]

    dst_root = data_root + '_new'
    for file in tqdm(image_files):
        if flag in file:
            dst_imp = file.replace(data_root, dst_root)
            os.makedirs(os.path.dirname(dst_imp), exist_ok=True)
            shutil.move(file, dst_imp)


if __name__ == '__main__':
    path = '/Users/lee/Desktop/jiaonanposhun/good'
    #B0:back
    flag = '-B1-'
    mode_data(path, flag)
