from imgaug import augmenters as iaa
from torchvision import transforms

from augment import transforms as aug_transforms


def config_aug(input_h, input_w, augment_set):
    # 进行模型初始化
    augment = {
        'flipleftright': iaa.Fliplr(0.5),
        'flipupdown': iaa.Flipud(0.5),
        'changecolorspace': iaa.ChangeColorspace(['BGR', 'GRAY', 'CIE', ], 'RGB', (0, 1)),
        'crop': iaa.Crop(percent=([0, 0.05], [0, 0.05], [0, 0.05], [0, 0.05])),
        'contrastnormalization': iaa.ContrastNormalization((0.5, 1.5)),
        'gaussiannoise': iaa.AdditiveGaussianNoise(0, (0, 5), 0.5),
        'multiply': iaa.Multiply((0.8, 1.2)),
        'scale': iaa.Affine(translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)}),
        'rotate': iaa.Affine(rotate=(-5, 5)),
        'shear': iaa.Affine(shear=(-5, 5)),
        'hueandsaturation': iaa.AddToHueAndSaturation((-10, 10)),
        'grayscale': iaa.Grayscale(alpha=(0, 0.5)),
    }

    children = []
    if augment_set is None:
        return children
    else:
        for aug in augment_set:
            method = augment.get(aug.lower())
            if method:
                children.append(method)

    iaa_aug_seq = iaa.Sequential(
        [iaa.SomeOf((0, None), children, random_order=True), iaa.Resize({"height": input_h, "width": input_w})])

    # 彩色图使用此值
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=mean,
                                     std=std)
    normalize_transpose = aug_transforms.NormalizeTranspose(mean=mean, std=std)
    return iaa_aug_seq, normalize, normalize_transpose
