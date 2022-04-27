import json
import math
import os
import sys
from datetime import datetime
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing
import torch.multiprocessing as mp
from absl import flags
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from config import get_default_config
from models.siamese_baseline import SiameseBaselineModelv1,SiameseLocalandMotionModelBIG, TwoBranchModel, ThreeBranchModel
from utils import TqdmToLogger, get_logger,AverageMeter,accuracy,ProgressMeter
from datasets import CityFlowNLDataset
from datasets import CityFlowNLInferenceDataset
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import time
import torch.nn.functional as F
from transformers import BertTokenizer,RobertaTokenizer
import pickle
from collections import OrderedDict
from utils import MgvSaveHelper


ossSaver = MgvSaveHelper()


def get_motion_texts(texts):
    direction = 0
    num_left = 0
    num_right = 0
    location = 0
    for sent in texts:
        if "intersection" in sent:
            location = 1
        if 'turn' in sent:
            if "left" in sent:
                num_left += 1
            if "right" in sent:
                num_right += 1

    if num_left > num_right:
        direction = 1
    if num_left < num_right:
        direction = 2

    motion_texts = []
    for idx in range(len(texts)):
        text = texts[idx]
        car_text = text.split('.')[0]

        motion_text = car_text
        if direction == 0:
            motion_text = motion_text + ' goes straight'
        if direction == 1:
            motion_text = motion_text + ' turns left'
        if direction == 2:
            motion_text = motion_text + ' turns right'
        if location == 1:
            if location == 1:
                if direction == '0':
                    motion_text = motion_text + ' through the intersection.'
                else:
                    motion_text = motion_text + ' at the intersection.'
        motion_texts.append(motion_text)
    return motion_texts


def get_motion_aug_text(texts):
    direction = 0
    num_left = 0
    num_right = 0
    location = 0
    for sent in texts:
        if "intersection" in sent:
            location = 1
        if 'turn' in sent:
            if "left" in sent:
                num_left += 1
            if "right" in sent:
                num_right += 1

    if num_left > num_right:
        direction = 1
    if num_left < num_right:
        direction = 2

    texts_aug = []
    for idx in range(len(texts)):
        ori_text = texts[idx].split('.')[1]
        if direction == 0:
            text = 'straight. ' + ori_text
        if direction == 1:
            text = 'left. ' + ori_text
        if direction == 2:
            text = 'right. ' + ori_text
        texts_aug.append(text)
    return texts_aug


parser = argparse.ArgumentParser(description='AICT5 Training')
parser.add_argument('--config', default="configs/baseline.yaml", type=str,
                    help='config_file')
parser.add_argument('--save-name', type=str, default="")

args = parser.parse_args()
out = dict()
use_cuda = True
cfg = get_default_config()
cfg.merge_from_file(args.config)
transform_test = torchvision.transforms.Compose([
    torchvision.transforms.Resize((cfg.DATA.SIZE, cfg.DATA.SIZE)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

ossSaver.set_stauts(save_oss=True, oss_path=cfg.DATA.OSS_PATH)

save_dir = "output"
os.makedirs(save_dir, exist_ok=True)

if args.save_name:
    save_name = args.save_name
else:
    save_name = args.config.split('/')[-1].split('.')[0]

if cfg.MODEL.NAME == "base":
    model = SiameseBaselineModelv1(cfg.MODEL)
elif cfg.MODEL.NAME == "dual-stream":
    model = SiameseLocalandMotionModelBIG(cfg.MODEL)
elif cfg.MODEL.NAME == 'two-branch':
    model = TwoBranchModel(cfg.MODEL)
elif cfg.MODEL.NAME == 'three-branch':
    model = ThreeBranchModel(cfg.MODEL)
else:
    raise NotImplementedError('model {} not implemented.'.format(cfg.MODEL.NAME))

# checkpoint = torch.load(cfg.TEST.RESTORE_FROM)
checkpoint = ossSaver.load_pth(ossSaver.get_s3_path(cfg.TEST.RESTORE_FROM))
new_state_dict = OrderedDict()
for k, v in checkpoint['state_dict'].items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v
model.load_state_dict(new_state_dict, strict=False)

if use_cuda:
    model.cuda()
    torch.backends.cudnn.benchmark = True

test_data = CityFlowNLInferenceDataset(cfg.DATA, transform=transform_test)
testloader = DataLoader(dataset=test_data, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False, num_workers=8)

if cfg.MODEL.BERT_TYPE == "BERT":
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
elif cfg.MODEL.BERT_TYPE == "ROBERTA":
    tokenizer = RobertaTokenizer.from_pretrained(cfg.MODEL.BERT_NAME)

model.eval()

# Extract Image Features
with torch.no_grad():
    for batch_idx, (image, motion, track_id, frames_id) in tqdm(enumerate(testloader)):
        vis_embed = model.encode_images(image.cuda(), motion.cuda())
        for i in range(len(track_id)):
            if track_id[i] not in out:
                out[track_id[i]] = dict()
            out[track_id[i]][frames_id[i].item()] = vis_embed[i,:].data.cpu().numpy()

pickle.dump(out, open(save_dir+'/img_feat_%s.pkl' % save_name, 'wb'))

# Extract Language Features
with open(cfg.TEST.QUERY_JSON_PATH) as f:
    queries = json.load(f)

query_embed = dict()

if cfg.MODEL.NAME == 'two-branch':
    with torch.no_grad():
        for q_id in tqdm(queries):
            car_texts = [text.split('.')[0] for text in queries[q_id]["nl"]]
            car_tokens = tokenizer.batch_encode_plus(car_texts, padding='longest', return_tensors='pt')
            texts = queries[q_id]["nl"]
            if cfg.DATA.MOTION_AUG:
                texts = get_motion_aug_text(texts)
            motion_tokens = tokenizer.batch_encode_plus(texts, padding='longest', return_tensors='pt')
            # tokens = tokenizer.batch_encode_plus([' '.join(queries[q_id]["nl"])], padding='longest', return_tensors='pt')
            lang_embeds = model.encode_text(motion_tokens['input_ids'].cuda(), motion_tokens['attention_mask'].cuda(), car_tokens['input_ids'].cuda(), car_tokens['attention_mask'].cuda())
            query_embed[q_id] = lang_embeds.data.cpu().numpy()
    pickle.dump(query_embed, open(save_dir + '/lang_feat_%s.pkl' % save_name, 'wb'))
elif cfg.MODEL.NAME == 'three-branch':
    with torch.no_grad():
        for q_id in tqdm(queries):
            car_texts = [text.split('.')[0] for text in queries[q_id]["nl"]]
            motion_texts = get_motion_texts(queries[q_id]["nl"])
            texts = queries[q_id]["nl"]
            if cfg.DATA.MOTION_AUG:
                texts = get_motion_aug_text(texts)
            car_tokens = tokenizer.batch_encode_plus(car_texts, padding='longest', return_tensors='pt')
            all_tokens = tokenizer.batch_encode_plus(texts, padding='longest', return_tensors='pt')
            motion_tokens = tokenizer.batch_encode_plus(motion_texts, padding='longest', return_tensors='pt')
            lang_embeds = model.encode_text(motion_tokens['input_ids'].cuda(), motion_tokens['attention_mask'].cuda(), car_tokens['input_ids'].cuda(), car_tokens['attention_mask'].cuda(), motion_tokens['input_ids'].cuda(), motion_tokens['attention_mask'].cuda())
            query_embed[q_id] = lang_embeds.data.cpu().numpy()
    pickle.dump(query_embed, open(save_dir + '/lang_feat_%s.pkl' % save_name, 'wb'))
else:
    with torch.no_grad():
        for q_id in tqdm(queries):
            texts = queries[q_id]["nl"]
            if cfg.DATA.MOTION_AUG:
                texts = get_motion_aug_text(texts)
            tokens = tokenizer.batch_encode_plus(texts, padding='longest', return_tensors='pt')
            # tokens = tokenizer.batch_encode_plus([' '.join(queries[q_id]["nl"])], padding='longest', return_tensors='pt')
            lang_embeds = model.encode_text(tokens['input_ids'].cuda(), tokens['attention_mask'].cuda())
            query_embed[q_id] = lang_embeds.data.cpu().numpy()
    pickle.dump(query_embed, open(save_dir+'/lang_feat_%s.pkl' % save_name, 'wb'))
