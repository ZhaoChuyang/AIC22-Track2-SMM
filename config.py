from yacs.config import CfgNode as CN

_C = CN()



# DATA process related configurations.
_C.DATA = CN()
_C.DATA.CITYFLOW_PATH = "/data/datasets/aicity2022/track2"
_C.DATA.TRAIN_JSON_PATH = "data/train.json"
_C.DATA.EVAL_JSON_PATH = "data/val.json"
_C.DATA.SIZE = 288
_C.DATA.CROP_AREA = 1. ## new_w = CROP_AREA * old_w
_C.DATA.TEST_TRACKS_JSON_PATH = "data/test_tracks.json"
_C.DATA.USE_MOTION = True
_C.DATA.MOTION_PATH = "data/motion_map"
_C.DATA.DATASET = "motion"  # ["motion", "paf"]
_C.DATA.OSS_PATH = "s3://zhaochuyang/"
_C.DATA.MOTION_AUG = False
_C.DATA.PID_INFO = False

# Model specific configurations.
_C.MODEL = CN()

_C.MODEL.NAME = "base"  # ["base", "dual-stream", "paf", "two-branch"]
_C.MODEL.BERT_TYPE = "BERT"
_C.MODEL.BERT_NAME = "bert-base-uncased"
_C.MODEL.IMG_ENCODER = "se_resnext50_32x4d"  # se_resnext50_32x4d, efficientnet-b2, efficientnet-b3
_C.MODEL.NUM_CLASS = 2498
_C.MODEL.NUM_CAMERAS = 40
_C.MODEL.EMBED_DIM = 1024
_C.MODEL.car_idloss = True
_C.MODEL.mo_idloss = True
_C.MODEL.share_idloss = True
_C.MODEL.camera_idloss = False
_C.MODEL.direction_loss = False
_C.MODEL.location_loss = False
_C.MODEL.RESNET_CHECKPOINT = ""
_C.MODEL.REID_PRETRAIN_WEIGHT = ""
_C.MODEL.BERT_PRETRAIN_WEIGHT = ""

# Training configurations
_C.TRAIN = CN()
_C.TRAIN.ONE_EPOCH_REPEAT = 30
_C.TRAIN.EPOCH = 40
_C.TRAIN.BATCH_SIZE = 64
_C.TRAIN.NUM_WORKERS = 4
_C.TRAIN.PRINT_FREQ = 20
_C.TRAIN.LR = CN()
_C.TRAIN.LR.BASE_LR = 0.01
_C.TRAIN.LR.WARMUP_EPOCH = 40
_C.TRAIN.LR.DELAY = 8
_C.TRAIN.EVAL_PERIOD = 10
_C.TRAIN.NUM_INSTANCE = 4

# Test configurations
_C.TEST = CN()
_C.TEST.RESTORE_FROM = None
_C.TEST.QUERY_JSON_PATH = "data/test_queries.json"
_C.TEST.BATCH_SIZE = 128
_C.TEST.NUM_WORKERS = 6
_C.TEST.CONTINUE = ""


def get_default_config():
    return _C.clone()