from huggingface_hub import HfApi

api = HfApi()


# # Track classification model, trained with MJN dataset
# api.create_repo(repo_id="LongshenOu/lead-inst-track-cls-mjn", repo_type="model", exist_ok=True)  # 确保仓库存在
# api.upload_file(
#     path_or_fileobj="/data2/longshen/lead_inst_detect_data/data/results/mjn_5pfm/chcls/chattn/ch_permute/bs16_anneallr_ep2_lnposattn/lightning_logs/version_0/checkpoints/epoch=00-valid_f1=0.8056-valid_loss=0.5538.ckpt",
#     path_in_repo="track_cls_mjn.ckpt",
#     repo_id="LongshenOu/lead-inst-track-cls-mjn",
#     repo_type="model"
# )

# # Track classification model, trained with MJN and MedleyDB dataset 
# repo_name = "LongshenOu/lead-inst-track-cls-mjn-medleydb"
# api.create_repo(repo_id=repo_name, repo_type="model", exist_ok=True)  # 确保仓库存在
# api.upload_file(
#     path_or_fileobj="/data2/longshen/lead_inst_detect_data/data/results/mjn_5pfm/cross_dataset/train_with_both/bs4_ep5_lr1e-4/lightning_logs/version_0/checkpoints/epoch=01-valid_f1=0.7320-valid_loss=0.6036.ckpt",
#     path_in_repo="track_cls_mjn_medleydb.ckpt",
#     repo_id=repo_name,
#     repo_type="model"
# )

# # Track classification model, trained with MedleyDB dataset 
# repo_name = "LongshenOu/lead-inst-track-cls-medleydb"
# api.create_repo(repo_id=repo_name, repo_type="model", exist_ok=True)  # 确保仓库存在
# api.upload_file(
#     path_or_fileobj="/data2/longshen/lead_inst_detect_data/data/results/mjn_5pfm/cross_dataset/train_with_medley/bs4_ep5_lr1e-4/lightning_logs/version_1/checkpoints/epoch=00-valid_f1=0.7527-valid_loss=0.7348.ckpt",
#     path_in_repo="track_cls_medleydb.ckpt",
#     repo_id=repo_name,
#     repo_type="model"
# )

# Segment-level guitar solo classification model, trained with MJN dataset 
repo_name = "LongshenOu/guitar-solo-segment-cls-mjn"
api.create_repo(repo_id=repo_name, repo_type="model", exist_ok=True)  # 确保仓库存在
api.upload_file(
    path_or_fileobj="/data2/longshen/lead_inst_detect_data/data/results/mjn_5pfm/instcls/from_mix_guitar_segment/lightning_logs/version_4/checkpoints/epoch=03-valid_f1=0.8217-valid_loss=0.3015.ckpt",
    path_in_repo="segment_guitar_cls_mjn.ckpt",
    repo_id=repo_name,
    repo_type="model"
)