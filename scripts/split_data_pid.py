import json


with open('data/train_tracks_pid.json', 'r') as f:
    train_annotation = json.load(f)

with open('data/train.json', 'r') as f:
    train = json.load(f)

with open('data/val.json', 'r') as f:
    valid = json.load(f)

for k in train.keys():
    train[k]['pid'] = train_annotation[k]['pid']

for k in valid.keys():
    valid[k]['pid'] = train_annotation[k]['pid']

with open('data/train_pid.json', 'w') as f:
    json.dump(train, f)

with open('data/val_pid.json', 'w') as f:
    json.dump(valid, f)
