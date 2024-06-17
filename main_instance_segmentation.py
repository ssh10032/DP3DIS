import wandb
wandb.init(project="mask3d")
import logging
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from hashlib import md5
from uuid import uuid4
import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from trainer.trainer import InstanceSegmentation, RegularCheckpointing
from pytorch_lightning.callbacks import ModelCheckpoint
from utils.utils import (
    flatten_dict,
    load_baseline_model,
    load_checkpoint_with_missing_or_exsessive_keys,
    load_backbone_checkpoint_with_missing_or_exsessive_keys
)
from pytorch_lightning import Trainer, seed_everything


def get_parameters(cfg: DictConfig):
    logger = logging.getLogger(__name__)
    load_dotenv(".env")

    # parsing input parameters
    seed_everything(cfg.general.seed)

    # getting basic configuration
    if cfg.general.get("gpus", None) is None:
        cfg.general.gpus = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    loggers = []

    # cfg.general.experiment_id = "0" # str(Repo("./").commit())[:8]
    # params = flatten_dict(OmegaConf.to_container(cfg, resolve=True))

    # create unique id for experiments that are run locally
    # unique_id = "_" + str(uuid4())[:4]
    # cfg.general.version = md5(str(params).encode("utf-8")).hexdigest()[:8] + unique_id

    if not os.path.exists(cfg.general.save_dir):
        os.makedirs(cfg.general.save_dir)
    else:
        print("EXPERIMENT ALREADY EXIST")
        cfg['trainer']['resume_from_checkpoint'] = f"{cfg.general.save_dir}/last-epoch.ckpt"

    for log in cfg.logging:
        print(log)
        loggers.append(hydra.utils.instantiate(log))
        loggers[-1].log_hyperparams(
            flatten_dict(OmegaConf.to_container(cfg, resolve=True))
        )

    model = InstanceSegmentation(cfg)
    if cfg.general.backbone_checkpoint is not None:
        cfg, model = load_backbone_checkpoint_with_missing_or_exsessive_keys(cfg, model)
    if cfg.general.checkpoint is not None:
        cfg, model = load_checkpoint_with_missing_or_exsessive_keys(cfg, model)

    logger.info(flatten_dict(OmegaConf.to_container(cfg, resolve=True)))
    return cfg, model, loggers


@hydra.main(config_path="conf", config_name="config_base_instance_segmentation.yaml")
def train(cfg: DictConfig):
    os.chdir(hydra.utils.get_original_cwd())
    cfg, model, loggers = get_parameters(cfg)
    callbacks = []
    for cb in cfg.callbacks:
        callbacks.append(hydra.utils.instantiate(cb))

    callbacks.append(RegularCheckpointing())

    runner = Trainer(
        logger=loggers,
        gpus=cfg.general.gpus,
        callbacks=callbacks,
        weights_save_path=str(cfg.general.save_dir),
        **cfg.trainer,
    )
    runner.fit(model)


@hydra.main(config_path="conf", config_name="config_base_instance_segmentation.yaml")
def test(cfg: DictConfig):
    # because hydra wants to change dir for some reason
    os.chdir(hydra.utils.get_original_cwd())
    cfg, model, loggers = get_parameters(cfg)
    runner = Trainer(
        gpus=cfg.general.gpus,
        logger=loggers,
        weights_save_path=str(cfg.general.save_dir),
        **cfg.trainer
    )
    runner.test(model)

@hydra.main(config_path="conf", config_name="config_base_instance_segmentation.yaml")
def main(cfg: DictConfig):
    # if cfg['general']['train_mode']:
    #     train(cfg)
    # else:
    #     test(cfg)
    ###############Train branch###############
    # cfg['general']['experiment_name'] = 'dn bbox_test'

    # cfg['general']['experiment_name'] = 'Songoh Ex PE Refinement V4 epoch 1000 batch 3'
    # cfg['general']['experiment_name'] = 'dn_nobbox_3d_coord_test'
    # cfg['general']['experiment_name'] = 'dn_nobbox_3d_coord_test_noise_0.3_fix_dnquery30'
    # cfg['general']['experiment_name'] = 'dn_nobbox_3d_coord_test_noise_0.3_fix_dnquery25'
    # cfg['general']['train_mode'] = True
    # cfg['general']['eval_on_segments'] = True
    # cfg['general']['train_on_segments'] = True
    # train(cfg)

    ###############Test branch###############
    cfg['general']['train_mode'] = False
    # best
    cfg['general']['checkpoint'] = 'saved/Query_selection_test/epoch=599-val_mean_ap_50=0.641.ckpt'
    # cfg['general']['checkpoint'] = 'saved/dn_nobbox_3d_coord_test_noise_0.3_fix_dnquery25/last-epoch.ckpt'
    cfg['general']['eval_on_segments'] = True
    cfg['general']['train_on_segments'] = True
    cfg['general']['use_dbscan'] = True
    cfg['general']['topk_per_image'] = -1
    cfg['general']['dbscan_eps'] = 0.95
    cfg['general']['save_visualizations'] = False
    cfg['general']['filter_out_instances'] = True
    cfg['general']['scores_threshold'] = 0.8
    test(cfg)



if __name__ == "__main__":
    main()
