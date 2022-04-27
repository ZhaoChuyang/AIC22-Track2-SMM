# Symmetric Network with Spatial Relationship Modeling for Natural Language-based Vehicle Retrieval

The 4nd place solution for AICity2022 Challenge track2: Natural Language-Based Vehicle Retrieval.

![framework](https://raw.githubusercontent.com/hbchen121/AICITY2022_Track2_SSM/master/imgs/framework.png)

We have two codebases and get the final results with these two:

1. One is this repo: https://github.com/ZhaoChuyang/AIC22-Track2-SMM
2. Another is at here: https://github.com/hbchen121/AICITY2022_Track2_SSM


## Prepare
Preprocess the dataset to prepare `frames, motion maps, NLP augmentation`

1. Run `python3 scripts/extract_vdo_frms.py` to extract frames from dataset.

2. Run `python3 scripts/deal_nlpaug.py` to perform NLP subject augmentation.

Generate post-processing features
1. Run `python3 scripts/get_location_info.py` to generate location information for each camera, which will be used in our post-processing stage.

2. Run `python3 scripts/get_relation_info.py` to generate relationship features for test tracks, which will be used in our post-processing stage.

## Train
Train model using the following configuration `configs/two_branch_cam_loc_dir.yaml`:

```
python -u main.py \
--name tb_cam_loc_dir \
--config configs/two_branch_cam_loc_dir.yaml
```

## Inference
Run `python test.py --config configs/two_branch_cam_loc_dir.yaml --save-name "tb_model"` to get test features.


## Post-Processing & Submit
Run `scripts/get_sumbmit.py` to get submitted file, post-processing is added by default.


## Others

If you have any questions, please leave an issue or contact us: [cy.zhao15@gmail.com](mailto:cy.zhao15@gmail.com) or [hbchen121@gmail.com](mailto:hbchen121@gmail.com).
