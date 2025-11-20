import math
import os
import random
import sys
import time
from datetime import datetime
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import configargparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from data.csi_datasets_indoor import get_loader
from config_ntc import config_ntc
from statics import evaluator
from utils import *


class RateDistortionLoss(nn.Module):
    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.nmse = evaluator
        self.lmbda = lmbda

    def forward(self, output, target, num_pixels=None):
        N, C, H, W = target.size()
        out = {}
        if num_pixels is None:
            num_pixels = N * H * W
        out["bpp"] = torch.log(output["likelihoods"]["w_res"]).sum() / (-math.log(2) * num_pixels)
        out["mse_loss_ntc"] = self.mse(output["x_hat"], target)
        out["mse_y"] = self.mse(output["y"], output["y_hat"])
        # out["loss"] = 0.01 * 255 ** 2 * (out["mse_loss_ntc"]) + self.lmbda * out["bpp"]
        out["loss"] = 1024 * (out["mse_loss_ntc"]) + out["bpp"]
        out["nmse"] = self.nmse(output["y"], output["y_hat"])
        return out


def train_one_epoch(epoch, net, criterion, train_loader, test_loader, optimizer_G, aux_optimizer,
                    lr_scheduler, device, logger):
    best_loss = float("inf")
    elapsed, losses, bpps, psnr_ntcs, y_mses = [AverageMeter() for _ in range(5)]
    metrics = [elapsed, losses, bpps, psnr_ntcs, y_mses]
    global global_step
    batch_idx = 0
    for input_image in train_loader:
        batch_idx += 1
        net.train()
        input_image = input_image[0].to(device)
        optimizer_G.zero_grad()
        #aux_optimizer.zero_grad()
        global_step += 1
        start_time = time.time()

        results = net(input_image)
        out_criterion = criterion(results, input_image)
        out_criterion["loss"].backward()

        if config.clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), config.clip_max_norm)
        optimizer_G.step()

        # aux_loss = net.aux_loss()
        # aux_loss.backward()
        # aux_optimizer.step()

        elapsed.update(time.time() - start_time)

        losses.update(out_criterion["loss"].item())
        bpps.update(out_criterion["bpp"].item())

        psnr_ntc = 10 * (torch.log(1. / out_criterion["mse_loss_ntc"]) / np.log(10))
        psnr_ntcs.update(psnr_ntc.item())
        y_mses.update(out_criterion["mse_y"].item())

        losses.update(out_criterion["loss"].item())

        if (global_step % config.print_every) == 0:
            process = (global_step % train_loader.__len__()) / (train_loader.__len__()) * 100.0
            log_info = [
                f'Step [{global_step % train_loader.__len__()}/{train_loader.__len__()}={process:.2f}%]',
                f'Loss {losses.val:.3f} ({losses.avg:.3f})',
                f'Time {elapsed.avg:.2f}',
                f'PSNR {psnr_ntcs.val:.2f} ({psnr_ntcs.avg:.2f})',
                f'Bpp_y {bpps.val:.2f} ({bpps.avg:.2f})',
                f'MSE_y {y_mses.val:.8f} ({y_mses.avg:.8f})',
                f'Epoch {epoch}'
            ]
            print(out_criterion["nmse"].item())
            log = (' | '.join(log_info))
            logger.info(log)
            if config.wandb:
                log_dict = {"Bpp_y": bpps.avg, "loss": losses.avg, "Step": global_step // config.print_every,
                            "PSNR_NTC": psnr_ntcs.avg}
                wandb.log(log_dict)
            for i in metrics:
                i.clear()

        lr_scheduler.step()

        if (global_step + 1) % config.test_every == 0:
            loss = test(net, criterion, test_loader, device, logger)
            is_best = loss < best_loss
            best_loss = min(loss, best_loss)
            if is_best:
                save_checkpoint(
                    {
                        "global_step": global_step,
                        "state_dict": net.state_dict(),
                        "optimizer": optimizer_G.state_dict(),
                        #"aux_optimizer": aux_optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                    },
                    is_best,
                    workdir
                )

        if (global_step + 1) % config.save_every == 0:
            save_checkpoint(
                {
                    "global_step": global_step,
                    "state_dict": net.state_dict(),
                    "optimizer": optimizer_G.state_dict(),
                    #"aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                False,
                workdir,
                filename='EP{}_{}.pth.tar'.format(epoch, global_step)
            )


def test(net, criterion, test_loader, device, logger):
    with torch.no_grad():
        net.eval()
        if not config.test_entropy_estimation:
            net.update()
        elapsed, losses, bst_losses, bpps, psnr_ntcs, y_mses = [AverageMeter() for _ in range(6)]
        print(len(test_loader))
        batch_idx = 0
        y_list = []
        w_list = []
        for input_image in test_loader:
            start_time = time.time()
            input_image = input_image[0].to(device)
            num_pixels = input_image.shape[0] * input_image.shape[2] * input_image.shape[3]
            if config.test_entropy_estimation:
                output = net(input_image)
                w_res_likelihoods = output["likelihoods"]['w_res']
                bpp = torch.log(w_res_likelihoods).sum() / (-math.log(2) * num_pixels)
                mse = criterion.mse(output["x_hat"], input_image)
                mse_y = criterion.mse(output["y"], output["y_hat"])
                loss = 0.01 * 255 ** 2 * mse + config.train_lambda * (bpp)
            else:
                out_enc = net.compress(input_image)
                y_string = out_enc["strings"][0]
                string_recon = y_string
                shape = out_enc["shape"]

                out_dec = net.decompress(string_recon, shape)
                bpp = sum(len(s) for s in y_string) * 8.0 / num_pixels


                mse = criterion.mse(out_dec["x_hat"], input_image)
                mse_y = criterion.mse(output["y"], output["y_hat"])
                loss = 0.01 * 255 ** 2 * mse + config.train_lambda * (bpp)
            elapsed.update(time.time() - start_time)
            losses.update(loss.item())
            bpps.update(bpp)
            y_mses.update(mse_y.item())
            psnr_ntc = 10 * (torch.log(1. / mse) / np.log(10))
            psnr_ntcs.update(psnr_ntc.item())
            log = (' | '.join([
                f'Step [{(batch_idx + 1)}/{test_loader.__len__()}]',
                f'Loss {losses.val:.3f} ({losses.avg:.3f})',
                f'Time {elapsed.avg:.2f}',
                f'PSNR {psnr_ntcs.val:.2f} ({psnr_ntcs.avg:.2f})',
                f'Bpp_y {bpps.val:.2f} ({bpps.avg:.2f})',
                f'MSE_y {y_mses.val:.8f} ({y_mses.avg:.8f})',
            ]))
            logger.info(log)
            batch_idx += 1

            if config.test_only:
                y_list.append(output["y"].cpu().numpy())
                w_list.append(output["w_res"].cpu().numpy())

        if config.test_only:
            B, C, w, h = y_list[2].shape
            y_np = np.array(y_list).transpose(2, 0, 1, 3, 4).reshape(C, -1, w*h)
            w_np = np.array(w_list).transpose(2, 0, 1, 3, 4).reshape(C, -1, w*h)
            y_mean = y_np.mean(axis=1)
            y_var = y_np.var(axis=1)
            y_std = y_np.std(axis=1)
            y_cov_matrix = cal_cov(y_np)
            w_mean = w_np.mean(axis=1)
            w_std = w_np.std(axis=1)
            w_var = w_np.var(axis=1)
            w_cov_matrix = cal_cov(w_np)
            w_mean_learned = net.mean.cpu().numpy().reshape(C, w*h)
            w_std_learned = net.scale.cpu().numpy().reshape(C, w*h)
            H_matrix = net.H.cpu().numpy()
            save_data = {
                "y_mean":y_mean,
                "y_cov_matrix": y_cov_matrix,
                "w_mean": w_mean,
                "w_std": w_std,
                "w_cov_matrix": w_cov_matrix,
                "w_mean_learned": w_mean_learned,
                "w_std_learned": w_std_learned,
                "H_matrix_learned":H_matrix
            }
            if not os.path.exists(workdir):
                os.makedirs(workdir)
            np.save(workdir + "/data4analysis.npy", save_data)
            C=4
            loaded_data = np.load(workdir + "/data4analysis.npy", allow_pickle=True).item()
            map1 = np.abs(y_cov_matrix[C])
            map2 = np.abs(w_cov_matrix[C])
            map3 = np.clip(y_cov_matrix[C]/w_cov_matrix[C], -3 , 3)
            # map1 = np.log10(np.clip(np.abs(y_cov_matrix[C]), 10**(-6), 10**(-2)))
            # map2 = np.log10(np.clip(np.abs(w_cov_matrix[C]),  10**(-6), 10**(-2)))
            # map3 = np.log10(np.clip(y_cov_matrix[C]/w_cov_matrix[C],  10**(-0.7), 10**(0.7)))
            plot_heatmap(map1, map2, map3,  workdir + "/heatmap.png")
            plot_mean_std(w_mean_learned[C], w_std_learned[C], w_mean[C], w_std[C], y_mean[C], y_std[C], workdir + "/mean_std.png")
            y_entropy = np.sum(0.5 * np.log(2 * np.pi * np.e * y_var), axis=1)
            w_entropy = np.sum(0.5 * np.log(2 * np.pi * np.e * w_var), axis=1)
            w_entropy_learned = np.sum(0.5 * np.log(2 * np.pi * np.e * (w_std_learned**2)), axis=1)
            plot_entropy(y_entropy, w_entropy, w_entropy_learned, workdir + "/entropy_w_y.png")

    if not config.test_only and config.wandb:
        wandb.log({"[Kodak] PSNR_NTC": psnr_ntcs.avg,
                   "[Kodak] Bpp_y": bpps.avg,
                   "[Kodak] loss": bst_losses.avg})
    return bst_losses.avg

class config1:
    seed = 1024
    CUDA = True
    device = torch.device("cuda:0")
    norm = False
    trainset = 'csiDataset_indoor'
    base_path = '/media/D/liangzijian/'
    data_dir = '/media/D/liangzijian/csi/%s/' % trainset

    use_side_info = True
    train_lambda = [1e3]                      # true_lambda = 1024

    # logger
    print_step = 50
    plot_step = 1000
    filename = datetime.now().__str__()[:-7]
    workdir = '/media/D/liangzijian/csi_flatFramework/angDelay_v3/history_indoors/NTC/lambda_1e3/{}'.format(filename)
    os.makedirs(workdir)
    log = []
    epoch_log = []
    for i in train_lambda:
        log.append((workdir + '/Log_lambda%06d.log' % i))
        epoch_log.append((workdir + '/epochLog_lambda%06d.csv' % i))
    samples = workdir + '/samples'
    models = workdir + '/models'
    os.makedirs(samples)
    os.makedirs(models)
    logger = None

    distortion_metric = 'MSE'

    embed_dim = 128

    # NTC training details
    # csi_dims = (2, 1024, 32)
    # csi_dims = (2, 128, 256)
    csi_dims = (2, 32, 32)
    normalize = False
    lr = {
        "base": 0.0001,        # 0.0001,
        "decay": 0.1,
        "decay_interval": 920000
    }

    warmup_step = 1000
    tot_step = 2500000
    tot_epoch = 1000
    save_model_freq = 40000
    test_step = 500000
    batch_size = 256                  # 1 * 200
    miniBatch_size = 100
    miniBatch_num = 200 // miniBatch_size

    eta = 0.2
    pass_channel = True
    channel = {"type": 'awgn', 'chan_param': 10}
    # multiple_rate = [16, 32, 48, 64, 80, 96, 102, 118, 134, 160, 186, 192, 208, 224, 240, 256]
    multiple_rate = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 72, 80, 88, 96, 104, 112, 120, 128, 144, 160, 176, 192, 208, 224, 240, 256]
    # multiple_rate = []

    ga_kwargs = dict(
        img_size=(csi_dims[1], csi_dims[2]), in_chans=2,
        embed_dims=[embed_dim, embed_dim, embed_dim], depths=[1, 1, 1], num_heads=[8, 8, 8],
        window_size=4, mlp_ratio=8., qkv_bias=True, qk_scale=None,
        norm_layer=nn.LayerNorm, patch_norm=True,
    )

    gs_kwargs = dict(
        img_size=(csi_dims[1], csi_dims[2]), out_chans=2,
        embed_dims=[embed_dim, embed_dim, embed_dim], depths=[1, 1, 1], num_heads=[8, 8, 8],
        window_size=4, mlp_ratio=8., norm_layer=nn.LayerNorm, patch_norm=True
    )

    # ha_kwargs = dict(
    #     input_resolution=(csi_dims[1] // 8), out_dim=embed_dim,
    #     embed_dim=embed_dim, depths=[1, 1, 1], num_heads=[8, 8, 8],
    #     window_size=4, mlp_ratio=4., qkv_bias=True, qk_scale=None,
    #     drop_rate=0., attn_drop_rate=0., drop_path_rate=0, norm_layer=nn.LayerNorm
    # )
    #
    # hs_kwargs = dict(
    #     input_resolution=(csi_dims[1] // 8), out_dim=embed_dim,
    #     embed_dim=embed_dim, depths=[1, 1, 1], num_heads=[8, 8, 8],
    #     window_size=4, mlp_ratio=4., qkv_bias=True, qk_scale=None,
    #     drop_rate=0., attn_drop_rate=0., drop_path_rate=0, norm_layer=nn.LayerNorm
    # )

    fe_kwargs = dict(
        input_resolution=(csi_dims[1] // 8),
        embed_dim=embed_dim, depths=[4], num_heads=[8],
        window_size=4, mlp_ratio=4., qkv_bias=True, qk_scale=None,
        norm_layer=nn.LayerNorm, rate_choice=multiple_rate
    )

    fd_kwargs = dict(
        input_resolution=(csi_dims[1] // 8),
        embed_dim=embed_dim, depths=[4], num_heads=[8],
        window_size=4, mlp_ratio=4., qkv_bias=True, qk_scale=None,
        norm_layer=nn.LayerNorm, rate_choice=multiple_rate
    )

def main(argv):
    global config
    config = config_ntc
    import torch.optim as optim
    from net.ntc import NTC


    if config.seed is not None:
        torch.manual_seed(config.seed)
        random.seed(config.seed)

    device = "cuda"
    config.device = device
    job_type = 'test' if config.test_only else 'train'
    exp_name = "[{}_lmbda={}] ".format(config.method, config.train_lambda) + config.exp_name
    global workdir
    workdir, logger = logger_configuration(exp_name, job_type,
                                           method=config.method, save_log=(not config.test_only))
    config.logger = logger
    logger.info(config.__dict__)

    net = NTC(N=config.N, M=config.M).to(device)

    criterion = RateDistortionLoss(lmbda=config.train_lambda)
    train_loader, test_loader = get_loader(config1)
    print(len(train_loader), len(test_loader))
    if config.test_only:
        if config.pretrained is not None:
            load_weights(net, config.pretrained)
        else:
            raise ValueError("Please specify the checkpoint path for testing.")
        test(net, criterion, test_loader, device, logger)

    else:
        if config.wandb:
            wandb_init_kwargs = {
                'project': 'NTC',
                'name': exp_name,
                'save_code': True,
                'job_type': job_type
            }
            wandb.init(**wandb_init_kwargs)

        G_params = set(p for n, p in net.named_parameters() if not n.endswith(".quantiles"))
        aux_params = set(p for n, p in net.named_parameters() if n.endswith(".quantiles"))
        optimizer_G = optim.Adam(G_params, lr=config.lr)
        aux_optimizer = None #optim.Adam(aux_params, lr=config.aux_lr)
        tot_epoch = config.epochs
        global global_step
        global_step = 0

        if config.pretrained is not None:
            global_step = 0
            pre_dict = torch.load(config.pretrained, map_location=device)
            result_dict = {}
            for key, weight in pre_dict["state_dict"].items():
                result_key = key
                if 'mask' not in key:
                    result_dict[result_key] = weight
            net.load_state_dict(result_dict, strict=False)
        elif config.checkpoint is not None and config.checkpoint != 'None':
            global_step = load_checkpoint(logger, device, global_step, net, optimizer_G, aux_optimizer,
                                          config.checkpoint)
        elif config.pretrained_ntc is not None:
            net.load_pretrained_ntc(config.pretrained_ntc)

        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer_G, milestones=[config.milestones], gamma=0.1)

        steps_epoch = global_step // train_loader.__len__()
        for epoch in range(steps_epoch, tot_epoch):
            logger.info('======Current epoch %s ======' % epoch)
            logger.info(f"Learning rate: {optimizer_G.param_groups[0]['lr']}")
            train_one_epoch(epoch, net, criterion, train_loader, test_loader, optimizer_G, aux_optimizer,
                            lr_scheduler, device, logger)


if __name__ == '__main__':
    main(sys.argv[1:])
