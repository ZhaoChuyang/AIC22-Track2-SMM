import json
import os
import random
from PIL import Image
import cv2
import torch
from torch.utils.data import Dataset, Sampler
import torch.nn.functional as F
import torchvision
from utils import get_logger
import pickle
import numpy as np
import math
from IPython import embed
import copy
import collections


class TripletSampler(Sampler):
    def __init__(self, data_source, batch_size, num_instances=4):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = batch_size // num_instances
        self.index_dict = dict()
        for idx, (uuid, pid) in enumerate(data_source):
            if pid not in self.index_dict:
                self.index_dict[pid] = [idx]
            else:
                self.index_dict[pid].append(idx)

        self.pids = list(self.index_dict.keys())
        self.length = len(data_source) // num_instances

    def __iter__(self):
        batch_idxs_dict = collections.defaultdict(list)
        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dict[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        self.length = len(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        return self.length


class CityFlowNLDataset(Dataset):
    def __init__(self, data_cfg, json_path, transform=None, Random=True):
        """
        Dataset for training.
        :param data_cfg: CfgNode for CityFlow NL.
        """

        self.data_cfg = data_cfg.clone()
        self.crop_area = data_cfg.CROP_AREA
        self.motion_aug = data_cfg.MOTION_AUG
        self.random = Random
        assert "nlpaug" in json_path
        with open(json_path) as f:
            tracks = json.load(f)
        self.list_of_uuids = list(tracks.keys())
        self.list_of_tracks = list(tracks.values())
        self.transform = transform
        self.bk_dic = {}
        self._logger = get_logger()
        
        self.all_indexs = list(range(len(self.list_of_uuids)))
        self.flip_tag = [False]*len(self.list_of_uuids)

        self.pid_info = False
        if 'pid' in json_path:
            self.pid_info = True
        print(json_path)
        if self.pid_info:
            all_pids = []
            for track in self.list_of_tracks:
                all_pids.append(track['pid'][0])
            all_pids = set(all_pids)

            self.pid2label = dict()
            for i, pid in enumerate(all_pids):
                self.pid2label[pid] = i

        self.cam2label = dict()
        for i in range(40):
            camera = 'c%03d' % (i+1)
            self.cam2label[camera] = i-1

        flip_aug = False
        if flip_aug:
            for i in range(len(self.list_of_uuids)):
                text = self.list_of_tracks[i]["nl"]
                for j in range(len(text)):
                    nl = text[j]
                    if "turn" in nl:
                        if "left" in nl:
                            self.all_indexs.append(i)
                            self.flip_tag.append(True)
                            break
                        elif "right" in nl:
                            self.all_indexs.append(i)
                            self.flip_tag.append(True)
                            break
        if self.pid_info:
            self.train = []

            for uuid, track in tracks.items():
                pid = self.pid2label[track["pid"][0]]
                self.train.append((uuid, pid))

        print(len(self.all_indexs))
        print("data load")

    def __len__(self):
        return len(self.all_indexs)

    def __getitem__(self, index):
        tmp_index = self.all_indexs[index]
        uuid = self.list_of_uuids[index]
        flag = self.flip_tag[index]
        track = self.list_of_tracks[tmp_index]
        if self.pid_info:
            pid = self.pid2label[track["pid"][0]]
        else:
            pid = 0

        if self.random:
            nl_idx = int(random.uniform(0, 3))
            frame_idx = int(random.uniform(0, len(track["frames"])))
        else:
            nl_idx = 2
            frame_idx = 0

        direction = 0
        num_left = 0
        num_right = 0
        location = 0
        for sent in track["nl"]:
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

        text = track["nl"][nl_idx]
        if self.motion_aug:
            ori_text = text.split('.')[1]
            if direction == 0:
                text = 'straight. ' + ori_text
            if direction == 1:
                text = 'left. ' + ori_text
            if direction == 2:
                text = 'right. ' + ori_text

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

        if flag:
            text = text.replace("left", "888888").replace("right", "left").replace("888888", "right")

        frame_path = os.path.join(self.data_cfg.CITYFLOW_PATH, track["frames"][frame_idx])
        frame = Image.open(frame_path)
        box = track["boxes"][frame_idx]

        camera = frame_path.split('/')[-3]
        camera_id = self.cam2label[camera]

        if self.crop_area == 1.6666667:
            box = (int(box[0]-box[2]/3.), int(box[1]-box[3]/3.), int(box[0]+4*box[2]/3.), int(box[1]+4*box[3]/3.))
        else:
            box = (int(box[0]-(self.crop_area-1)*box[2]/2.), int(box[1]-(self.crop_area-1)*box[3]/2), int(box[0]+(self.crop_area+1)*box[2]/2.), int(box[1]+(self.crop_area+1)*box[3]/2.))

        crop = frame.crop(box)
        if flag:
            crop = torch.flip(crop, [1])
        crop_data = self.transform(crop)
        frame.close()

        if self.data_cfg.USE_MOTION:
            bk_path = self.data_cfg.MOTION_PATH+"/%s.jpg" % self.list_of_uuids[tmp_index]
            bk = Image.open(bk_path)
            bk_data = self.transform(bk)
            bk.close()
            return {
                "uuid": uuid,
                "crop_data": crop_data,
                "text": text,
                "car_text": car_text,
                "bk_data": bk_data,
                "tmp_index": tmp_index,
                "camera_id": camera_id,
                "direction": direction,
                "location_id": location,
                "motion_text": motion_text,
                "pid": pid,
            }

        return {
            "uuid": uuid,
            "crop_data": crop_data,
            "text": text,
            "car_text": car_text,
            "tmp_index": tmp_index,
            "camera_id": camera_id,
            "direction": direction,
            "location_id": location,
            "motion_text": motion_text,
            "pid": pid,
        }


class CityFlowNLInferenceDataset(Dataset):
    def __init__(self, data_cfg,transform = None):
        """Dataset for evaluation. Loading tracks instead of frames."""
        self.data_cfg = data_cfg
        self.crop_area = data_cfg.CROP_AREA
        self.transform = transform

        with open(self.data_cfg.TEST_TRACKS_JSON_PATH) as f:
            tracks = json.load(f)
        self.list_of_uuids = list(tracks.keys())
        self.list_of_tracks = list(tracks.values())
        self.list_of_crops = list()
        for track_id_index, track in enumerate(self.list_of_tracks):
            for frame_idx, frame in enumerate(track["frames"]):
                frame_path = os.path.join(self.data_cfg.CITYFLOW_PATH, frame)
                box = track["boxes"][frame_idx]
                crop = {"frame": frame_path, "frames_id":frame_idx,"track_id": self.list_of_uuids[track_id_index], "box": box}
                self.list_of_crops.append(crop)
        self._logger = get_logger()


    def __len__(self):
        return len(self.list_of_crops)

    def __getitem__(self, index):
        track = self.list_of_crops[index]
        frame_path = track["frame"]

        box = track["box"]
        if self.crop_area == 1.6666667:
            box = (int(box[0]-box[2]/3.),int(box[1]-box[3]/3.), int(box[0]+4*box[2]/3.),int(box[1]+4*box[3]/3.))
        else:
            box = (int(box[0]-(self.crop_area-1)*box[2]/2.),int(box[1]-(self.crop_area-1)*box[3]/2),int(box[0]+(self.crop_area+1)*box[2]/2.),int(box[1]+(self.crop_area+1)*box[3]/2.))

        frame = Image.open(frame_path)
        crop = frame.crop(box)
        frame.close()

        if self.transform is not None:
            crop = self.transform(crop)
        if self.data_cfg.USE_MOTION:
            bk = Image.open(self.data_cfg.MOTION_PATH+"/%s.jpg" % track["track_id"])
            bk_data = self.transform(bk)
            bk.close()
            return crop, bk_data, track["track_id"], track["frames_id"]
        return crop, track["track_id"], track["frames_id"]


class CityFlowNLDatasetv2(Dataset):
    def __init__(self, data_cfg, json_path, transform=None, Random=True, pid_info=False, transform_paf= None):
        """
        Dataset for training.
        :param data_cfg: CfgNode for CityFlow NL.
        """
        self.data_cfg = data_cfg.clone()
        self.crop_area = data_cfg.CROP_AREA
        self.random = Random
        with open(json_path) as f:
            tracks = json.load(f)
        self.list_of_uuids = list(tracks.keys())
        self.list_of_tracks = list(tracks.values())
        self.transform = transform
        self.transform_paf = transform_paf

        self.paf_width = 40

        self._logger = get_logger()

        self.all_indexs = list(range(len(self.list_of_uuids)))
        self.flip_tag = [False] * len(self.list_of_uuids)

        self.pid_info = pid_info
        if self.pid_info:
            all_pids = []
            for track in self.list_of_tracks:
                all_pids.append(track['pid'][0])
            all_pids = set(all_pids)
            self.pid2label = dict()
            for i, pid in enumerate(all_pids):
                self.pid2label[pid] = i

        flip_aug = False
        if flip_aug:
            for i in range(len(self.list_of_uuids)):
                text = self.list_of_tracks[i]["nl"]
                for j in range(len(text)):
                    nl = text[j]
                    if "turn" in nl:
                        if "left" in nl:
                            self.all_indexs.append(i)
                            self.flip_tag.append(True)
                            break
                        elif "right" in nl:
                            self.all_indexs.append(i)
                            self.flip_tag.append(True)
                            break
        print(len(self.all_indexs))
        print("data load")

    def __len__(self):
        return len(self.all_indexs)

    def set_paf_map(self, paf_map: np.array, x_a: int, y_a: int, x_b: int, y_b: int, width: int = 50) -> None:
        x_ba = x_b - x_a
        y_ba = y_b - y_a
        h_map, w_map, _ = paf_map.shape

        x_min = int(max(min(x_a, x_b) - width, 0))
        x_max = int(min(max(x_a, x_b) + width, w_map))
        y_min = int(max(min(y_a, y_b) - width, 0))
        y_max = int(min(max(y_a, y_b) + width, h_map))

        norm_ba = math.sqrt((x_ba * x_ba + y_ba * y_ba))
        if norm_ba < 1e-7:
            return
        x_ba = x_ba / norm_ba
        y_ba = y_ba / norm_ba

        for y in range(y_min, y_max):
            for x in range(x_min, x_max):
                x_ca = x - x_a
                y_ca = y - y_a
                d = math.fabs(x_ca * y_ba - y_ca * x_ba)
                if d <= width:
                    if paf_map[y, x, 0] > 1e-7:
                        paf_map[y, x, 0] = (x_ba + paf_map[y, x, 0]) / 2
                        paf_map[y, x, 1] = (y_ba + paf_map[y, x, 1]) / 2
                    else:
                        paf_map[y, x, 0] = x_ba
                        paf_map[y, x, 1] = y_ba

    def generate_paf_map(self, frames: list, bboxes: list, root: str) -> np.array:
        centroids = []
        for box in bboxes:
            x = int(box[0] + 1 / 2 * box[2])
            y = int(box[1] + 1 / 2 * box[3])
            centroids.append((x, y))

        height, width, _ = cv2.imread(os.path.join(root, frames[0])).shape
        paf_map = np.zeros((height, width, 2))

        assert len(centroids) == len(frames)

        for i in range(len(frames) - 1):
            p1 = centroids[i]
            p2 = centroids[i + 1]
            self.set_paf_map(paf_map, p1[0], p1[1], p2[0], p2[1], self.paf_width)

        return paf_map

    def __getitem__(self, index):
        tmp_index = self.all_indexs[index]
        flag = self.flip_tag[index]
        track = self.list_of_tracks[tmp_index]
        uuid = self.list_of_uuids[tmp_index]

        if self.random:
            nl_idx = int(random.uniform(0, 3))
            frame_idx = int(random.uniform(0, len(track["frames"])))
        else:
            nl_idx = 2
            frame_idx = 0
        text = track["nl"][nl_idx]
        if flag:
            text = text.replace("left", "888888").replace("right", "left").replace("888888", "right")

        frame_path = os.path.join(self.data_cfg.CITYFLOW_PATH, track["frames"][frame_idx])

        frame = Image.open(frame_path)

        # frame.load()
        box = track["boxes"][frame_idx]
        if self.crop_area == 1.6666667:
            box = (int(box[0] - box[2] / 3.), int(box[1] - box[3] / 3.), int(box[0] + 4 * box[2] / 3.),
                   int(box[1] + 4 * box[3] / 3.))
        else:
            box = (int(box[0] - (self.crop_area - 1) * box[2] / 2.), int(box[1] - (self.crop_area - 1) * box[3] / 2),
                   int(box[0] + (self.crop_area + 1) * box[2] / 2.), int(box[1] + (self.crop_area + 1) * box[3] / 2.))

        crop = frame.crop(box)
        crop_data = self.transform(crop)

        if self.data_cfg.USE_MOTION:
            # paf_path = os.path.join(self.data_cfg.PAF_PATH, self.list_of_uuids[tmp_index] + '.pkl')
            # with open(paf_path, 'rb') as fb:
            #     paf_map = pickle.load(fb)
            # paf_map = self.generate_paf_map(track['frames'], track['boxes'], self.data_cfg.CITYFLOW_PATH)
            with open('data/paf_maps/%s.pkl' % uuid, 'rb') as fb:
                paf_map = pickle.load(fb)
            # print(paf_map.shape)
            paf_map = self.transform_paf(paf_map)
            # print(paf_map.shape)
            frame_img = np.array(frame)
            # print(frame_img.shape)
            frame_img = self.transform(frame_img)
            # print(frame_img.shape)
            motion_img = torch.cat([frame_img, paf_map], dim=0).float()
            return crop_data, text, motion_img, tmp_index
        if flag:
            crop = torch.flip(crop, [1])
        # frame.close()
        return crop_data, text, tmp_index


class CityFlowNLInferenceDatasetv2(Dataset):
    def __init__(self, data_cfg,transform = None, transform_paf= None):
        """Dataset for evaluation. Loading tracks instead of frames."""
        self.data_cfg = data_cfg
        self.crop_area = data_cfg.CROP_AREA
        self.transform = transform
        with open(self.data_cfg.TEST_TRACKS_JSON_PATH) as f:
            tracks = json.load(f)
        self.list_of_uuids = list(tracks.keys())
        self.list_of_tracks = list(tracks.values())
        self.list_of_crops = list()
        self.paf_width = 40
        self.transform_paf = transform_paf

        for track_id_index,track in enumerate(self.list_of_tracks):
            for frame_idx, frame in enumerate(track["frames"]):
                frame_path = os.path.join(self.data_cfg.CITYFLOW_PATH, frame)
                box = track["boxes"][frame_idx]
                crop = {"frame": frame_path, "frames_id":frame_idx,"track_id": self.list_of_uuids[track_id_index], "box": box, "frames": track["frames"], "boxes": track["boxes"]}
                self.list_of_crops.append(crop)
        self._logger = get_logger()

    def __len__(self):
        return len(self.list_of_crops)

    def set_paf_map(self, paf_map: np.array, x_a: int, y_a: int, x_b: int, y_b: int, width: int = 50) -> None:
        x_ba = x_b - x_a
        y_ba = y_b - y_a
        h_map, w_map, _ = paf_map.shape

        x_min = int(max(min(x_a, x_b) - width, 0))
        x_max = int(min(max(x_a, x_b) + width, w_map))
        y_min = int(max(min(y_a, y_b) - width, 0))
        y_max = int(min(max(y_a, y_b) + width, h_map))

        norm_ba = math.sqrt((x_ba * x_ba + y_ba * y_ba))
        if norm_ba < 1e-7:
            return
        x_ba = x_ba / norm_ba
        y_ba = y_ba / norm_ba

        for y in range(y_min, y_max):
            for x in range(x_min, x_max):
                x_ca = x - x_a
                y_ca = y - y_a
                d = math.fabs(x_ca * y_ba - y_ca * x_ba)
                if d <= width:
                    if paf_map[y, x, 0] > 1e-7:
                        paf_map[y, x, 0] = (x_ba + paf_map[y, x, 0]) / 2
                        paf_map[y, x, 1] = (y_ba + paf_map[y, x, 1]) / 2
                    else:
                        paf_map[y, x, 0] = x_ba
                        paf_map[y, x, 1] = y_ba

    def generate_paf_map(self, frames: list, bboxes: list, root: str) -> np.array:
        centroids = []
        for box in bboxes:
            x = int(box[0] + 1 / 2 * box[2])
            y = int(box[1] + 1 / 2 * box[3])
            centroids.append((x, y))

        height, width, _ = cv2.imread(os.path.join(root, frames[0])).shape
        paf_map = np.zeros((height, width, 2))

        assert len(centroids) == len(frames)

        for i in range(len(frames) - 1):
            p1 = centroids[i]
            p2 = centroids[i + 1]
            self.set_paf_map(paf_map, p1[0], p1[1], p2[0], p2[1], self.paf_width)

        return paf_map

    def __getitem__(self, index):
        track = self.list_of_crops[index]
        frame_path = track["frame"]

        box = track["box"]
        uuid = track['track_id']
        if self.crop_area == 1.6666667:
            box = (int(box[0]-box[2]/3.),int(box[1]-box[3]/3.),int(box[0]+4*box[2]/3.),int(box[1]+4*box[3]/3.))
        else:
            box = (int(box[0]-(self.crop_area-1)*box[2]/2.),int(box[1]-(self.crop_area-1)*box[3]/2),int(box[0]+(self.crop_area+1)*box[2]/2.),int(box[1]+(self.crop_area+1)*box[3]/2.))

        frame = Image.open(frame_path)
        crop = frame.crop(box)

        if self.transform is not None:
            crop = self.transform(crop)
        if self.data_cfg.USE_MOTION:
            # paf_path = os.path.join(self.data_cfg.PAF_PATH, tracks["track_id"] + '.pkl')
            # with open(paf_path, 'rb') as fb:
            #     paf_map = pickle.load(fb)
            # paf_map = self.generate_paf_map(track['frames'], track['boxes'], self.data_cfg.CITYFLOW_PATH)
            with open('data/paf_maps/%s.pkl' % uuid, 'rb') as fb:
                paf_map = pickle.load(fb)
            paf_map = self.transform_paf(paf_map)
            frame_img = np.array(frame)
            frame_img = self.transform(frame_img)
            motion_img = torch.cat([frame_img, paf_map], dim=0).float()
            return crop, motion_img, track["track_id"], track["frames_id"]
        return crop, track["track_id"], track["frames_id"]