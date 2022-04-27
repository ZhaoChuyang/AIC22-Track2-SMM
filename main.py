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
import numpy as np
from config import get_default_config
from models.siamese_baseline import SiameseBaselineModelv1, SiameseLocalandMotionModelBIG, SiameseRank2, SiamesePafModelBig, TwoBranchModel, TwoBranchRNN
from utils import TqdmToLogger, get_logger, AverageMeter, accuracy, ProgressMeter, TestLogger, get_lr, MgvSaveHelper
from datasets import CityFlowNLDataset, CityFlowNLDatasetv2, CityFlowNLInferenceDataset
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import time
import torch.nn.functional as F
from transformers import BertTokenizer, RobertaTokenizer, RobertaModel
from collections import OrderedDict
from torchmetrics import RetrievalMRR
import nltk


best_mrr_eval = 0.
best_mrr_test = 0.

mrr_metric = RetrievalMRR()
test_logger = TestLogger()
ossSaver = MgvSaveHelper()

glove = None


def get_lang_v(texts):
    location = 0
    direction = 0

    num_left = 0
    num_right = 0
    for text in texts:
        if 'turn' in text:
            if 'left' in text:
                num_left += 1
            if 'right' in text:
                num_right += 1
        if 'intersection' in text:
            location = 1

    if num_left > num_right:
        direction = 1
    if num_left < num_right:
        direction = 2

    loc_map = [[1,0], [0,1]]
    dir_map = [[1,0,0], [0,1,0], [0,0,1]]
    return loc_map[location], dir_map[direction]


def tokenize(sentences):
    all_tokens = []
    for sentence in sentences:
        sent_len = len(sentence)
        words = nltk.word_tokenize(sentence.lower())
        tokens = np.zeros((sent_len, 300), dtype=np.float32)
        for i, w in enumerate(words):
            if i >= sent_len:
                break
            try:
                tokens[i] = glove[w]
            except KeyError:
                pass
        all_tokens.append(tokens)
    return torch.tensor(all_tokens)


class WarmUpLR(_LRScheduler):
    def __init__(self, lr_scheduler, warmup_steps, eta_min=1e-7):
        self.lr_scheduler = lr_scheduler
        self.warmup_steps = warmup_steps
        self.eta_min = eta_min
        super().__init__(lr_scheduler.optimizer, lr_scheduler.last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [self.eta_min + (base_lr - self.eta_min) * (self.last_epoch / self.warmup_steps)
                    for base_lr in self.base_lrs]
        return self.lr_scheduler.get_lr()

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        if epoch < self.warmup_steps:
            super().step(epoch)
        else:
            self.last_epoch = epoch
            self.lr_scheduler.step(epoch - self.warmup_steps)

    def state_dict(self):
        state_dict = {key: value for key, value in self.__dict__.items() if key not in ('optimizer', 'lr_scheduler')}
        state_dict['lr_scheduler'] = self.lr_scheduler.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        lr_scheduler = state_dict.pop('lr_scheduler')
        self.__dict__.update(state_dict)
        self.lr_scheduler.load_state_dict(lr_scheduler)


# def evaluate(model, tokenizer, optimizer, valloader, epoch, cfg, save_dir, index=2):
#     global best_top1_eval
#     print("Test::::")
#     model.eval()
#     batch_time = AverageMeter('Time', ':6.3f')
#     data_time = AverageMeter('Data', ':6.3f')
#     losses = AverageMeter('Loss', ':6.3f')
#     top1_acc = AverageMeter('Acc@1', ':6.2f')
#     top5_acc = AverageMeter('Acc@5', ':6.2f')
#     mrr = AverageMeter('MRR', ':6.2f')
#
#     end = time.time()
#     with torch.no_grad():
#         for batch_idx, batch in enumerate(valloader):
#             if cfg.DATA.USE_MOTION:
#                 image, text, bk, id_car = batch
#             else:
#                 image, text, id_car = batch
#             tokens = tokenizer.batch_encode_plus(text, padding='longest', return_tensors='pt')
#             data_time.update(time.time() - end)
#             if cfg.DATA.USE_MOTION:
#                 pairs, logit_scale, cls_logits = model(tokens['input_ids'].cuda(), tokens['attention_mask'].cuda(), image.cuda(), bk.cuda())
#             else:
#                 pairs, logit_scale, cls_logits = model(tokens['input_ids'].cuda(), tokens['attention_mask'].cuda(), image.cuda())
#             logit_scale = logit_scale.mean().exp()
#             loss = 0
#
#             # for visual_embeds,lang_embeds in pairs:
#             visual_embeds, lang_embeds = pairs[index]
#             sim_i_2_t = torch.matmul(torch.mul(logit_scale, visual_embeds), torch.t(lang_embeds))
#             sim_t_2_i = sim_i_2_t.t()
#             loss_t_2_i = F.cross_entropy(sim_t_2_i, torch.arange(image.size(0)).cuda())
#             loss_i_2_t = F.cross_entropy(sim_i_2_t, torch.arange(image.size(0)).cuda())
#             loss += (loss_t_2_i+loss_i_2_t)/2
#
#             acc1, acc5 = accuracy(sim_t_2_i, torch.arange(image.size(0)).cuda(), topk=(1, 5))
#             mrr_score = mrr_metric(
#                 sim_t_2_i.flatten(),
#                 torch.eye(len(sim_t_2_i), device=sim_t_2_i.device).long().bool().flatten(),
#                 torch.arange(len(sim_t_2_i), device=sim_t_2_i.device)[:, None].expand(len(sim_t_2_i),
#                                                                                       len(sim_t_2_i)).flatten()
#             )
#
#             losses.update(loss.detach().item(), image.size(0))
#             top1_acc.update(acc1[0], image.size(0))
#             top5_acc.update(acc5[0], image.size(0))
#             mrr.update(mrr_score.detach().item(), image.size(0))
#
#             batch_time.update(time.time() - end)
#             end = time.time()
#
#             print(
#                 'Epoch: [{}][{}/{}]\t'
#                 'Time {:.3f} ({:.3f}) '
#                 'Data {:.3f} ({:.3f}) '
#                 'Loss {:.3f} ({:.3f}) '
#                 'Acc@1 {:.3f} ({:.3f}) '
#                 'Acc@5 {:.3f} ({:.3f}) '
#                 'MRR {:.2%} ({:.2%}) '.format(
#                     epoch, batch_idx, len(valloader),
#                     batch_time.val, batch_time.avg,
#                     data_time.val, data_time.avg,
#                     losses.val, losses.avg,
#                     top1_acc.val, top1_acc.avg,
#                     top5_acc.val, top5_acc.avg,
#                     mrr.val, mrr.avg
#                 )
#             )
#     record = [epoch, '%6.3f' % top1_acc.avg, '%6.3f' % top5_acc.avg, '%6.3f' % mrr.avg, '%6.6f' % get_lr(optimizer)]
#     test_logger.print_table(record)
#
#     if mrr.avg > best_top1_eval:
#         best_top1_eval = mrr.avg
#         checkpoint_file = save_dir + "/checkpoint_best_eval.pth"
#         torch.save(
#             {"epoch": epoch,
#              "state_dict": model.state_dict(),
#              "optimizer": optimizer.state_dict()}, checkpoint_file)

def evaluatev2(model, valloader, epoch, cfg, save_dir, index=[0, 1], save=True, tokenizer=None, optimizer=None):
    global best_mrr_eval
    print("====> Validation")
    model.eval()
    all_visual_embeds = [[] for _ in range(len(index)+1)]
    all_lang_embeds = [[] for _ in range(len(index)+1)]
    with torch.no_grad():
        for batch_idx, batch in enumerate(valloader):
            image = batch["crop_data"]
            text = batch["text"]
            car_text = batch["car_text"]
            id_car = batch["tmp_index"]
            cam_id = batch["camera_id"]
            if cfg.DATA.USE_MOTION:
                bk = batch["bk_data"]

            if glove:
                motion_tokens = tokenize(text)
                car_tokens = tokenize(text)
                outputs = model(motion_tokens, car_tokens, image.cuda(), bk.cuda())
            else:
                motion_tokens = tokenizer.batch_encode_plus(text, padding='longest', return_tensors='pt')
                car_tokens = tokenizer.batch_encode_plus(car_text, padding='longest', return_tensors='pt')
                outputs = model(motion_tokens['input_ids'].cuda(), motion_tokens['attention_mask'].cuda(), car_tokens['input_ids'], car_tokens['attention_mask'], image.cuda(), bk.cuda())

            pairs = outputs["pairs"]
            logit_scale = outputs["logit_scale"]
            cls_logits = outputs["cls_logits"]
            cam_preds = outputs["cam_preds"]

            logit_scale = logit_scale.mean().exp()

            # for visual_embeds, lang_embeds in pairs:
            for idx in index:
                visual_embeds, lang_embeds = pairs[idx]
                all_visual_embeds[idx].append(visual_embeds.detach())
                all_lang_embeds[idx].append(lang_embeds.detach())

    for idx in index:
        visual_embeds = torch.cat(all_visual_embeds[idx])
        lang_embeds = torch.cat(all_lang_embeds[idx])
        all_sim = torch.matmul(visual_embeds, torch.t(lang_embeds))
        sim_i_2_t = torch.mul(logit_scale, all_sim)
        sim_t_2_i = sim_i_2_t.t()

        acc1, acc5 = accuracy(all_sim, torch.arange(all_sim.size(0)).cuda(), topk=(1, 5))
        mrr_score = mrr_metric(
            sim_t_2_i.flatten(),
            torch.eye(len(sim_t_2_i), device=sim_t_2_i.device).long().bool().flatten(),
            torch.arange(len(sim_t_2_i), device=sim_t_2_i.device)[:, None].expand(len(sim_t_2_i),
                                                                                  len(sim_t_2_i)).flatten()
        )

        top1_acc = acc1[0]
        top5_acc = acc5[0]
        mrr = mrr_score.item()

        record = [f"{epoch} ( d-{idx} )", '%6.3f' % top1_acc, '%6.3f' % top5_acc, '%6.3f' % mrr, '%6.6f' % get_lr(optimizer)]
        test_logger.add_record(record)

    '''
    Concatenate All Features
    '''
    visual_embeds = []
    lang_embeds = []
    for idx in index:
        visual_embeds.append(torch.cat(all_visual_embeds[idx]))
        lang_embeds.append(torch.cat(all_lang_embeds[idx]))
    # visual_embeds: [2, num_tracks, embedding_dim]
    visual_embeds = torch.stack(visual_embeds, dim=0).mean(dim=0)
    lang_embeds = torch.stack(lang_embeds, dim=0).mean(dim=0)

    # visual_embeds: [num_tracks, embedding_dim*2]
    # visual_embeds = torch.cat(visual_embeds, dim=-1)
    # lang_embeds: [num_tracks, embedding_dim*2]
    # lang_embeds = torch.cat(lang_embeds, dim=-1)
    all_sim = torch.matmul(visual_embeds, torch.t(lang_embeds))
    sim_i_2_t = torch.mul(logit_scale, all_sim)
    sim_t_2_i = sim_i_2_t.t()

    acc1, acc5 = accuracy(all_sim, torch.arange(all_sim.size(0)).cuda(), topk=(1, 5))
    mrr_score = mrr_metric(
        sim_t_2_i.flatten(),
        torch.eye(len(sim_t_2_i), device=sim_t_2_i.device).long().bool().flatten(),
        torch.arange(len(sim_t_2_i), device=sim_t_2_i.device)[:, None].expand(len(sim_t_2_i), len(sim_t_2_i)).flatten()
    )

    top1_acc = acc1[0]
    top5_acc = acc5[0]
    mrr = mrr_score.item()

    record = [f"{epoch} ( d-mean )", '%6.3f' % top1_acc, '%6.3f' % top5_acc, '%6.3f' % mrr, '%6.6f' % get_lr(optimizer)]
    test_logger.add_record(record)

    visual_embeds = []
    lang_embeds = []
    for idx in index:
        visual_embeds.append(torch.cat(all_visual_embeds[idx]))
        lang_embeds.append(torch.cat(all_lang_embeds[idx]))
    # visual_embeds: [2, num_tracks, embedding_dim]
    # visual_embeds = torch.stack(visual_embeds, dim=0).mean(dim=0)
    # lang_embeds = torch.stack(lang_embeds, dim=0).mean(dim=0)

    # visual_embeds: [num_tracks, embedding_dim*2]
    visual_embeds = torch.cat(visual_embeds, dim=-1)
    # lang_embeds: [num_tracks, embedding_dim*2]
    lang_embeds = torch.cat(lang_embeds, dim=-1)
    all_sim = torch.matmul(visual_embeds, torch.t(lang_embeds))
    sim_i_2_t = torch.mul(logit_scale, all_sim)
    sim_t_2_i = sim_i_2_t.t()

    acc1, acc5 = accuracy(all_sim, torch.arange(all_sim.size(0)).cuda(), topk=(1, 5))
    mrr_score = mrr_metric(
        sim_t_2_i.flatten(),
        torch.eye(len(sim_t_2_i), device=sim_t_2_i.device).long().bool().flatten(),
        torch.arange(len(sim_t_2_i), device=sim_t_2_i.device)[:, None].expand(len(sim_t_2_i), len(sim_t_2_i)).flatten()
    )

    top1_acc = acc1[0]
    top5_acc = acc5[0]
    mrr = mrr_score.item()

    record = [f"{epoch} ( d-cat )", '%6.3f' % top1_acc, '%6.3f' % top5_acc, '%6.3f' % mrr, '%6.6f' % get_lr(optimizer)]
    # test_logger.print_table(record)
    test_logger.print_table(record)

    if mrr > best_mrr_eval and save:
        best_mrr_eval = mrr
        checkpoint_file = save_dir + "/checkpoint_best_eval.pth"
        ossSaver.save_ckpt(checkpoint_file, epoch, model, optimizer)


def evaluate_by_test_aug(model, valloader, epoch, cfg, save_dir, save=True, tokenizer=None, optimizer=None, test_aug_path=""):
    global best_mrr_test
    print(f"====> Test")
    model.eval()

    with open(test_aug_path, 'r') as fb:
        test_aug = json.load(fb)

    loc_logits = []
    dir_logits = []
    lang_loc_v = []
    lang_dir_v = []

    with torch.no_grad():

        all_visual_embeds = dict()
        out = dict()
        for batch_idx, (image, motion, track_id, frames_id) in tqdm(enumerate(valloader), total=len(valloader)):
            vis_embed = model.module.encode_images(image.cuda(), motion.cuda())
            for i in range(len(track_id)):
                if track_id[i] not in out:
                    out[track_id[i]] = dict()
                out[track_id[i]][frames_id[i].item()] = vis_embed[i, :].data.cpu()

        for track_id, img_feat in out.items():
            tmp = []
            for fid in img_feat:
                tmp.append(img_feat[fid])
            tmp = torch.stack(tmp)
            tmp = torch.mean(tmp, 0)
            all_visual_embeds[track_id] = tmp
            loc_logits[track_id] = F.softmax(test_aug[track_id]['loc'])
            dir_logits[track_id] = F.softmax(test_aug[track_id]['dir'])

        all_lang_embeds = dict()

        with open(cfg.DATA.EVAL_JSON_PATH) as f:
            print(f"====> Query {cfg.DATA.EVAL_JSON_PATH} load")
            queries = json.load(f)

        if cfg.MODEL.NAME == 'two-branch':
            for q_id in tqdm(queries):
                texts = queries[q_id]["nl"]
                car_texts = [text.split('.')[0] for text in queries[q_id]["nl"]]
                car_tokens = tokenizer.batch_encode_plus(car_texts, padding='longest', return_tensors='pt')
                motion_tokens = tokenizer.batch_encode_plus(queries[q_id]["nl"], padding='longest', return_tensors='pt')
                lang_embeds = model.module.encode_text(motion_tokens['input_ids'].cuda(),
                                                motion_tokens['attention_mask'].cuda(), car_tokens['input_ids'].cuda(),
                                                car_tokens['attention_mask'].cuda())
                all_lang_embeds[q_id] = lang_embeds.data.cpu()
                loc_v, dir_v = get_lang_v(texts)
                lang_loc_v[q_id] = loc_v
                lang_dir_v[q_id] = dir_v
        else:
            for q_id in tqdm(queries):
                tokens = tokenizer.batch_encode_plus(queries[q_id]["nl"], padding='longest', return_tensors='pt')
                # tokens = tokenizer.batch_encode_plus([' '.join(queries[q_id]["nl"])], padding='longest', return_tensors='pt')
                lang_embeds = model.module.encode_text(tokens['input_ids'].cuda(), tokens['attention_mask'].cuda())
                all_lang_embeds[q_id] = lang_embeds.data.cpu()

        all_sim = []
        visual_embeds = []
        loc_embeds = []
        dir_embeds = []
        for q_id in all_visual_embeds.keys():
            visual_embeds.append(all_visual_embeds[q_id])
            loc_embeds.append(loc_logits[q_id])
            dir_embeds.append(dir_logits[q_id])
        visual_embeds = torch.stack(visual_embeds)
        loc_embeds = torch.stack(loc_embeds)
        dir_embeds = torch.stack(dir_embeds)

        for q_id in tqdm(all_visual_embeds.keys()):
            '''
            cur_sim(torch): (1, num_tracks), similarity between lang[qid] and all_visual_embeds
            '''
            lang_embeds = all_lang_embeds[q_id]
            cur_sim = torch.mean(torch.matmul(lang_embeds, visual_embeds.T), 0, keepdim=True)
            all_sim.append(cur_sim)


        all_sim = torch.cat(all_sim)
        sim_t_2_i = all_sim
        sim_i_2_t = sim_t_2_i.t()

        acc1, acc5 = accuracy(all_sim, torch.arange(all_sim.size(0)), topk=(1, 5))
        mrr_score = mrr_metric(
            sim_t_2_i.flatten(),
            torch.eye(len(sim_t_2_i), device=sim_t_2_i.device).long().bool().flatten(),
            torch.arange(len(sim_t_2_i), device=sim_t_2_i.device)[:, None].expand(len(sim_t_2_i), len(sim_t_2_i)).flatten()
        )

        mrr = mrr_score.item() * 100
        top1_acc = acc1[0]
        top5_acc = acc5[0]

        record = [f"{epoch} ( test )", '%6.3f' % top1_acc, '%6.3f' % top5_acc, '%6.3f' % mrr, '%6.6f' % get_lr(optimizer)]
        test_logger.print_table(record)

        if mrr > best_mrr_test and save:
            best_mrr_test = mrr
            checkpoint_file = save_dir + "/checkpoint_best_test.pth"
            ossSaver.save_ckpt(checkpoint_file, epoch, model, optimizer)


def evaluate_by_test(model, valloader, epoch, cfg, save_dir, save=True, tokenizer=None, optimizer=None):
    global best_mrr_test
    print(f"====> Test")
    model.eval()
    with torch.no_grad():

        all_visual_embeds = dict()
        out = dict()
        for batch_idx, (image, motion, track_id, frames_id) in tqdm(enumerate(valloader), total=len(valloader)):
            vis_embed = model.module.encode_images(image.cuda(), motion.cuda())
            for i in range(len(track_id)):
                if track_id[i] not in out:
                    out[track_id[i]] = dict()
                out[track_id[i]][frames_id[i].item()] = vis_embed[i, :].data.cpu()

        for track_id, img_feat in out.items():
            tmp = []
            for fid in img_feat:
                tmp.append(img_feat[fid])
            tmp = torch.stack(tmp)
            tmp = torch.mean(tmp, 0)
            all_visual_embeds[track_id] = tmp

        all_lang_embeds = dict()

        with open(cfg.DATA.EVAL_JSON_PATH) as f:
            print(f"====> Query {cfg.DATA.EVAL_JSON_PATH} load")
            queries = json.load(f)

        if cfg.MODEL.NAME == 'two-branch':
            for q_id in tqdm(queries):
                car_texts = [text.split('.')[0] for text in queries[q_id]["nl"]]
                car_tokens = tokenizer.batch_encode_plus(car_texts, padding='longest', return_tensors='pt')
                motion_tokens = tokenizer.batch_encode_plus(queries[q_id]["nl"], padding='longest', return_tensors='pt')
                lang_embeds = model.module.encode_text(motion_tokens['input_ids'].cuda(),
                                                motion_tokens['attention_mask'].cuda(), car_tokens['input_ids'].cuda(),
                                                car_tokens['attention_mask'].cuda())
                all_lang_embeds[q_id] = lang_embeds.data.cpu()
        else:
            for q_id in tqdm(queries):
                tokens = tokenizer.batch_encode_plus(queries[q_id]["nl"], padding='longest', return_tensors='pt')
                # tokens = tokenizer.batch_encode_plus([' '.join(queries[q_id]["nl"])], padding='longest', return_tensors='pt')
                lang_embeds = model.module.encode_text(tokens['input_ids'].cuda(), tokens['attention_mask'].cuda())
                all_lang_embeds[q_id] = lang_embeds.data.cpu()

        all_sim = []
        visual_embeds = []
        for q_id in all_visual_embeds.keys():
            visual_embeds.append(all_visual_embeds[q_id])
        visual_embeds = torch.stack(visual_embeds)
        for q_id in tqdm(all_visual_embeds.keys()):
            lang_embeds = all_lang_embeds[q_id]
            cur_sim = torch.mean(torch.matmul(lang_embeds, visual_embeds.T), 0, keepdim=True)
            all_sim.append(cur_sim)

        all_sim = torch.cat(all_sim)
        sim_t_2_i = all_sim
        sim_i_2_t = sim_t_2_i.t()

        acc1, acc5 = accuracy(all_sim, torch.arange(all_sim.size(0)), topk=(1, 5))
        mrr_score = mrr_metric(
            sim_t_2_i.flatten(),
            torch.eye(len(sim_t_2_i), device=sim_t_2_i.device).long().bool().flatten(),
            torch.arange(len(sim_t_2_i), device=sim_t_2_i.device)[:, None].expand(len(sim_t_2_i), len(sim_t_2_i)).flatten()
        )

        mrr = mrr_score.item() * 100
        top1_acc = acc1[0]
        top5_acc = acc5[0]

        record = [f"{epoch} ( test )", '%6.3f' % top1_acc, '%6.3f' % top5_acc, '%6.3f' % mrr, '%6.6f' % get_lr(optimizer)]
        test_logger.print_table(record)

        if mrr > best_mrr_test and save:
            best_mrr_test = mrr
            checkpoint_file = save_dir + "/checkpoint_best_test.pth"
            ossSaver.save_ckpt(checkpoint_file, epoch, model, optimizer)


def evaluate(model, valloader, epoch, cfg, save_dir, index=-1, save=True, tokenizer=None, optimizer=None):
    global best_mrr_eval
    print("====> Validation")
    evl_start_time = time.monotonic()
    model.eval()
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Clip Loss', ':.4e')
    mrr = AverageMeter('MRR', ':6.4f')
    top1_acc = AverageMeter('Acc@1', ':6.4f')
    top5_acc = AverageMeter('Acc@5', ':6.4f')

    end = time.time()

    all_visual_embeds = []
    all_lang_embeds = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(valloader):
            image = batch["crop_data"]
            text = batch["text"]
            id_car = batch["tmp_index"]
            cam_id = batch["camera_id"]
            if cfg.DATA.USE_MOTION:
                bk = batch["bk_data"]

            tokens = tokenizer.batch_encode_plus(text, padding='longest', return_tensors='pt')
            data_time.update(time.time() - end)

            outputs = model(tokens['input_ids'].cuda(), tokens['attention_mask'].cuda(), image.cuda(), bk.cuda())
            pairs = outputs["pairs"]
            logit_scale = outputs["logit_scale"]
            cls_logits = outputs["cls_logits"]
            cam_preds = outputs["cam_preds"]

            logit_scale = logit_scale.mean().exp()

            # for visual_embeds, lang_embeds in pairs:
            visual_embeds, lang_embeds = pairs[index]
            all_visual_embeds.append(visual_embeds.detach())
            all_lang_embeds.append(lang_embeds.detach())

    all_visual_embeds = torch.cat(all_visual_embeds)
    all_lang_embeds = torch.cat(all_lang_embeds)
    all_sim = torch.matmul(all_visual_embeds, torch.t(all_lang_embeds))
    sim_i_2_t = torch.mul(logit_scale, all_sim)
    sim_t_2_i = sim_i_2_t.t()
    loss_t_2_i = F.cross_entropy(sim_t_2_i, torch.arange(sim_t_2_i.size(0)).cuda())
    loss_i_2_t = F.cross_entropy(sim_i_2_t, torch.arange(sim_t_2_i.size(0)).cuda())
    loss = (loss_t_2_i + loss_i_2_t) / 2

    acc1, acc5 = accuracy(all_sim, torch.arange(all_sim.size(0)).cuda(), topk=(1, 5))
    mrr_score = mrr_metric(
        sim_t_2_i.flatten(),
        torch.eye(len(sim_t_2_i), device=sim_t_2_i.device).long().bool().flatten(),
        torch.arange(len(sim_t_2_i), device=sim_t_2_i.device)[:, None].expand(len(sim_t_2_i),
                                                                              len(sim_t_2_i)).flatten()
    )

    losses.update(loss.item(), image.size(0))
    mrr.update(mrr_score.item(), image.size(0))
    top1_acc.update(acc1[0], image.size(0))
    top5_acc.update(acc5[0], image.size(0))
    batch_time.update(time.time() - end)

    evl_end_time = time.monotonic()
    print(
        'Epoch: [{}][{}/{}]\t'
        'Time {:.3f} ({:.3f}) '
        'Data {:.3f} ({:.3f}) '
        'Loss {:.3f} ({:.3f}) '
        'Acc@1 {:.3f} ({:.3f}) '
        'Acc@5 {:.3f} ({:.3f}) '
        'MRR {:.2%} ({:.2%}) '.format(
            epoch, batch_idx, len(valloader),
            batch_time.val, batch_time.avg,
            data_time.val, data_time.avg,
            losses.val, losses.avg,
            top1_acc.val, top1_acc.avg,
            top5_acc.val, top5_acc.avg,
            mrr.val, mrr.avg
        )
    )
    record = [epoch, '%6.3f' % top1_acc.avg, '%6.3f' % top5_acc.avg, '%6.3f' % mrr.avg, '%6.6f' % get_lr(optimizer)]
    test_logger.print_table(record)

    if mrr.avg > best_mrr_eval and save:
        best_mrr_eval = mrr.avg
        checkpoint_file = save_dir + "/checkpoint_best_eval.pth"
        ossSaver.save_ckpt(checkpoint_file, epoch, model, optimizer)


def main():
    parser = argparse.ArgumentParser(description='AICT5 Training')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--config', default="configs/baseline.yaml", type=str,
                        help='config_file')
    parser.add_argument('--name', default="baseline", type=str,
                        help='experiments')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--test-aug', default="", type=str)
    args = parser.parse_args()

    cfg = get_default_config()
    cfg.merge_from_file(args.config)

    print(cfg)

    ossSaver.set_stauts(save_oss=True, oss_path=cfg.DATA.OSS_PATH)

    transform_train = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.RandomResizedCrop(cfg.DATA.SIZE, scale=(0.8, 1.)),
        torchvision.transforms.RandomApply([torchvision.transforms.RandomRotation(10)], p=0.5),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((cfg.DATA.SIZE, cfg.DATA.SIZE)),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    use_cuda = True
    if cfg.DATA.DATASET == "motion":
        train_data = CityFlowNLDataset(cfg.DATA, json_path=cfg.DATA.TRAIN_JSON_PATH, transform=transform_test)
        val_data = CityFlowNLDataset(cfg.DATA, json_path=cfg.DATA.EVAL_JSON_PATH, transform=transform_test, Random=False)
        test_data = CityFlowNLInferenceDataset(cfg.DATA, transform=transform_test)
    elif cfg.DATA.DATASET == "paf":
        train_data = CityFlowNLDatasetv2(cfg.DATA, json_path=cfg.DATA.TRAIN_JSON_PATH, transform=transform_test)
        val_data = CityFlowNLDatasetv2(cfg.DATA, json_path=cfg.DATA.EVAL_JSON_PATH, transform=transform_test,
                                       Random=False)
    else:
        raise NotImplemented("Dataset {} not implemented".format(cfg.DATA.DATASET))

    trainloader = DataLoader(dataset=train_data, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True,
                             num_workers=cfg.TRAIN.NUM_WORKERS, pin_memory=False)
    valloader = DataLoader(dataset=val_data, batch_size=cfg.TRAIN.BATCH_SIZE * 20, shuffle=False,
                           num_workers=cfg.TRAIN.NUM_WORKERS, pin_memory=False)
    testloader = DataLoader(dataset=test_data, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False,
                            num_workers=cfg.TRAIN.NUM_WORKERS)

    if cfg.MODEL.NAME == "base":
        model = SiameseBaselineModelv1(cfg.MODEL)
    elif cfg.MODEL.NAME == "dual-stream":
        model = SiameseLocalandMotionModelBIG(cfg.MODEL)
    elif cfg.MODEL.NAME == "rank2":
        model = SiameseRank2(cfg.MODEL)
    elif cfg.MODEL.NAME == "paf":
        model = SiamesePafModelBig(cfg.MODEL)
    elif cfg.MODEL.NAME == "two-branch":
        model = TwoBranchModel(cfg.MODEL)
    else:
        assert "unsupported model"

    if args.resume or args.evaluate:
        checkpoint = ossSaver.load_pth(ossSaver.get_s3_path(cfg.TEST.RESTORE_FROM))
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

    if use_cuda:
        model.cuda()
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        # cudnn.benchmark = True

    # params = [
    #     {"params": model.module.vis_params(), "lr": cfg.TRAIN.LR.BASE_LR},
    #     {"params": model.module.lang_params(), "lr": cfg.TRAIN.LR.BASE_LR*0.001},
    # ]
    params = [{"params": model.parameters(), "lr": cfg.TRAIN.LR.BASE_LR}]

    optimizer = torch.optim.AdamW(params)
    step_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=len(trainloader)*cfg.TRAIN.LR.DELAY, gamma=0.1)
    scheduler = WarmUpLR(lr_scheduler=step_scheduler, warmup_steps=int(1.*cfg.TRAIN.LR.WARMUP_EPOCH*len(trainloader)))

    if cfg.MODEL.BERT_TYPE == "BERT":
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    elif cfg.MODEL.BERT_TYPE == "ROBERTA":
        tokenizer = RobertaTokenizer.from_pretrained(cfg.MODEL.BERT_NAME)
    else:
        tokenizer = None
        raise NotImplementedError('Tokenizer not implemented')
    global_step = 0
    best_top1 = 0.

    if args.evaluate:
        if cfg.MODEL.NAME == "two-branch":
            evaluatev2(model=model, valloader=valloader, epoch=0, cfg=cfg, save_dir=args.name, index=[0, 1],
                       save=False, tokenizer=tokenizer, optimizer=optimizer)
        else:
            evaluate(model=model, valloader=valloader, epoch=0, cfg=cfg, save_dir=args.name, index=-1, save=False,
                     tokenizer=tokenizer, optimizer=optimizer)

        evaluate_by_test(model=model, valloader=testloader, epoch=0, cfg=cfg, save_dir=args.name, save=False,
                         tokenizer=tokenizer, optimizer=optimizer)
        exit(0)

    for epoch in range(cfg.TRAIN.EPOCH):
        if epoch % cfg.TRAIN.EVAL_PERIOD == 0:
            if cfg.MODEL.NAME == "two-branch":
                evaluatev2(model=model, valloader=valloader, epoch=epoch, cfg=cfg, save_dir=args.name, index=[0, 1],
                           save=True, tokenizer=tokenizer, optimizer=optimizer)
            else:
                evaluate(model=model, valloader=valloader, epoch=epoch, cfg=cfg, save_dir=args.name, index=-1,
                         save=True, tokenizer=tokenizer, optimizer=optimizer)

        if epoch % (cfg.TRAIN.EVAL_PERIOD * 2) == 0 and epoch > 0:
            evaluate_by_test(model=model, valloader=testloader, epoch=epoch, cfg=cfg, save_dir=args.name, save=True,
                             tokenizer=tokenizer, optimizer=optimizer)

        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':6.3f')
        top1_acc = AverageMeter('Acc@1', ':6.2f')
        top5_acc = AverageMeter('Acc@5', ':6.2f')
        mrr = AverageMeter('MRR', ':6.2f')

        model.train()

        for batch_idx, batch in enumerate(trainloader):
            end = time.time()
            image = batch["crop_data"]
            text = batch["text"]
            car_text = batch["car_text"]
            id_car = batch["tmp_index"]
            cam_id = batch["camera_id"]
            direction_id = batch["direction"]
            location_id = batch["location_id"]
            if cfg.DATA.USE_MOTION:
                bk = batch["bk_data"]

            global_step += 1
            optimizer.zero_grad()

            motion_tokens = tokenizer.batch_encode_plus(text, padding='longest', return_tensors='pt')
            car_tokens = tokenizer.batch_encode_plus(car_text, padding='longest', return_tensors='pt')
            data_time.update(time.time() - end)

            if cfg.MODEL.NAME == 'two-branch':
                outputs = model(motion_tokens['input_ids'].cuda(), motion_tokens['attention_mask'].cuda(),
                                car_tokens['input_ids'], car_tokens['attention_mask'], image.cuda(), bk.cuda())
            else:
                outputs = model(motion_tokens['input_ids'].cuda(), motion_tokens['attention_mask'].cuda(), image.cuda(), bk.cuda())

            pairs = outputs["pairs"]
            logit_scale = outputs["logit_scale"]
            cls_logits = outputs["cls_logits"]
            cam_preds = outputs["cam_preds"]
            direction_preds = outputs["direction_preds"]
            location_preds = outputs["location_preds"]

            logit_scale = logit_scale.mean().exp()
            loss = 0.

            for visual_embeds, lang_embeds in pairs:
                sim_i_2_t = torch.matmul(torch.mul(logit_scale, visual_embeds), torch.t(lang_embeds))
                sim_t_2_i = sim_i_2_t.t()
                loss_t_2_i = F.cross_entropy(sim_t_2_i, torch.arange(image.size(0)).cuda())
                loss_i_2_t = F.cross_entropy(sim_i_2_t, torch.arange(image.size(0)).cuda())
                loss += (loss_t_2_i+loss_i_2_t)/2

            for cls_logit in cls_logits:
                loss += 0.5 * F.cross_entropy(cls_logit, id_car.long().cuda())

            if cfg.MODEL.camera_idloss:
                for preds in cam_preds:
                    loss += F.cross_entropy(preds, cam_id.cuda())

            if cfg.MODEL.direction_loss:
                for preds in direction_preds:
                    loss += F.cross_entropy(preds, direction_id.cuda())

            if cfg.MODEL.location_loss:
                for preds in location_preds:
                    loss += F.cross_entropy(preds, location_id.cuda())

            loss.backward()
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                acc1, acc5 = accuracy(sim_t_2_i, torch.arange(image.size(0)).cuda(), topk=(1, 5))

                mrr_score = mrr_metric(
                    sim_t_2_i.flatten(),
                    torch.eye(len(sim_t_2_i), device=sim_t_2_i.device).long().bool().flatten(),
                    torch.arange(len(sim_t_2_i), device=sim_t_2_i.device)[:, None].expand(len(sim_t_2_i),
                                                                                          len(sim_t_2_i)).flatten()
                )

                losses.update(loss.detach().item(), image.size(0))
                top1_acc.update(acc1[0], image.size(0))
                top5_acc.update(acc5[0], image.size(0))
                mrr.update(mrr_score.detach().item(), image.size(0))
                batch_time.update(time.time() - end)
                end = time.time()

            if batch_idx % cfg.TRAIN.PRINT_FREQ == 0:
                print(
                    'Epoch: [{}][{}/{}]\t'
                    'Time {:.3f} ({:.3f}) '
                    'Data {:.3f} ({:.3f}) '
                    'Loss {:.3f} ({:.3f}) '
                    'Acc@1 {:.3f} ({:.3f}) '
                    'Acc@5 {:.3f} ({:.3f}) '
                    'MRR {:.2%} ({:.2%}) '.format(
                        epoch, batch_idx, len(trainloader),
                        batch_time.val, batch_time.avg,
                        data_time.val, data_time.avg,
                        losses.val, losses.avg,
                        top1_acc.val, top1_acc.avg,
                        top5_acc.val, top5_acc.avg,
                        mrr.val, mrr.avg
                    )
                )

        if epoch % 20 == 1:
            checkpoint_file = args.name + "/checkpoint_%d.pth" % epoch
            ossSaver.save_ckpt(checkpoint_file, epoch, model, optimizer)
            # torch.save(
            #     {"epoch": epoch, "global_step": global_step,
            #      "state_dict": model.state_dict(),
            #      "optimizer": optimizer.state_dict()}, checkpoint_file)

        if mrr.avg > best_top1:
            best_top1 = mrr.avg
            checkpoint_file = args.name+"/checkpoint_best.pth"
            ossSaver.save_ckpt(checkpoint_file, epoch, model, optimizer)
            # torch.save(
            #     {"epoch": epoch, "global_step": global_step,
            #      "state_dict": model.state_dict(),
            #      "optimizer": optimizer.state_dict()}, checkpoint_file)


if __name__ == '__main__':
    main()
