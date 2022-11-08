from yacs.config import CfgNode as CN
import torch
__C = CN()
__C.project                             = 'demo'
__C.net                                 = ''
__C.task_type                           = ''

# Training config
# common args
__C.data_root                           = '/home/lee/data/shuangjing/data'
__C.input_w                             = 320
__C.input_h                             = 192
__C.classes                             = 11
__C.batch_size                          = 128
__C.test_batch_size                     = 10
__C.start_epoch                         = 0
__C.epochs = 50
__C.balance_n_classes                   = 8
__C.balance_n_samples                   = 16
__C.resume                              = ''
__C.pretrained                          = True
__C.device                              = 'gpu'
__C.workers                             = 8
__C.seed                                = 0
__C.lr                                  = 0.001
__C.threshold                           = 0.5
__C.gpus                                = 0
__C.topk                                = [1,2]
__C.print_freq                          = 10  # step
__C.save_freq                           = 500  # step
__C.criterion_list                      = ['CrossEntropyLoss']
__C.train_transforms                    = None
__C.val_transforms                      = None
__C.train_data_path                     = ''
__C.val_data_path                       = ''
# ext args
__C.class_weight                        = []
__C.period_weights                      = 0.0  # auto update
__C.period_thresh                       = 0.0  # auto update
__C.period_n_min                        = 0  # auto update
__C.cycle_train_times                   = 1
__C.area_threshold                      = 50
__C.train_outputs                       = [0]
__C.score_threshold                     = 0.1
__C.remove_no_json_sample_flag          = False
__C.convert_no_json_sample_flag         = False
__C.convert_float_flag                  = False
__C.class_id_map_save_path              = 'temp/class_id_map.txt'
__C.period_n_min                        = 10000
__C.asymmetry_id                        = False
__C.data_count                          = 0
__C.emb_template_path                   = ''
# Display & Save


# Data Augmentation Params
__C.augment = CN()
__C.augment.crop                        = False
__C.augment.change_colorspace           = False
__C.augment.contrast_normalization      = False
__C.augment.gaussian_noise              = False
__C.augment.multiply                    = False
__C.augment.scale                       = False
__C.augment.rotate                      = False
__C.augment.shear                       = False
__C.augment.grayscale                   = False
__C.augment.hue_and_saturation          = False
__C.augment.flip_left_right             = False
__C.augment.flip_up_down                = False
__C.augment.mean                        = [0.485, 0.456, 0.406]
__C.augment.std                         = [0.229, 0.224, 0.225]

def get_default_config_freeze():
    __C.freeze()
    return __C.clone()


def get_default_config():
    return __C.clone()


def merge_from_file(file):
    __C.merge_from_file(file)
    return __C.clone()


if __name__ == '__main__':
    df = get_default_config()
    print(df)
    print(type(df))
    print(df.augment.rotate)
