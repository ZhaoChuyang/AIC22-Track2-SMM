import math
import json
import numpy as np
import random

train_data = []
labels = []

intersection_dict = {}

with open("/data/AIC21-R1/data/train_tracks.json") as f:
    train_tracks = json.load(f)

with open("/data/AIC21-R1/data/test_tracks.json") as f:
    test_tracks = json.load(f)

cam1 = set()
cam2 = set()

for _, record in train_tracks.items():
    for frame in record['frames']:
        cam1.add(frame.split('/')[-3])

for _, record in test_tracks.items():
    for frame in record['frames']:
        cam2.add(frame.split('/')[-3])

for tgt_camera in cam2:
    is_intersection = 0
    for uuid, record in test_tracks.items():

        camera = record['frames'][0].split('/')[-3]

        if camera != tgt_camera:
            continue

        texts = record['nl']
        path = []
        for box in record['boxes']:
            x = box[0] + box[2] // 2
            y = box[1] + box[3] // 2
            path.append(np.array([x, y, 1]))

        gps_path = path

        n = len(gps_path)

        x = [p[0] for p in gps_path]
        y = [p[1] for p in gps_path]

        x = [v + random.random() * 0.001 for v in x]
        y = [v + random.random() * 0.001 for v in y]

        path = [(x, y) for x, y, _ in path]
        moves = [0] * len(path)
        for i in range(len(path) - 1):
            moves[i] = ((path[i + 1][0] - path[i][0]) ** 2 + (path[i + 1][1] - path[i][1]) ** 2) ** (1 / 2)

        num_stops = 0
        pre = 0

        if camera not in intersection_dict:
            intersection_dict[camera] = 0

        num_zeros = 0

        for v in moves:
            num_zeros = max(num_zeros, num_stops)
            if v < 1e-6:
                pre = 0
                num_stops += 1
            if v > 1e-6:
                pre = v
                num_stops = 0

        if num_stops >= 15:
            intersection_dict[camera] = 1

print(intersection_dict)

