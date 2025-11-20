import math
import os
import random
import sys
import time
from datetime import datetime
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import configargparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

#from data.datasets import get_cifar10_loader
from config_ntc import config_ntc
from data.csi_datasets_indoor import get_loader
from utils import logger_configuration, load_weights, load_checkpoint, AverageMeter, save_checkpoint


class RateDistortionLoss(nn.Module):
    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda

    def forward(self, output, target, num_pixels=None):
        N, C, H, W = target.size()
        out = {}
        if num_pixels is None:
            num_pixels = N * H * W
        out["bpp_y"] = torch.log(output["likelihoods"]["y"]).sum() / (-math.log(2) * num_pixels)
        out["bpp_z"] = torch.log(output["likelihoods"]["z"]).sum() / (-math.log(2) * num_pixels)
        out["mse_loss_ntc"] = self.mse(output["x_hat"], target)
        out["bpp_loss"] = out["bpp_y"] + out["bpp_z"]
        out["loss"] = 0.01 * 255 ** 2 * (out["mse_loss_ntc"]) + self.lmbda * out["bpp_loss"]
        return out


def train_one_epoch(epoch, net, criterion, train_loader, test_loader, optimizer_G, aux_optimizer,
                    lr_scheduler, device, logger):
    best_loss = float("inf")
    elapsed, losses, bppys, bppzs, psnr_ntcs = [AverageMeter() for _ in range(5)]
    metrics = [elapsed, losses, bppys, bppzs, psnr_ntcs]
    global global_step
    batch_idx = 0
    for input_image in train_loader:
        batch_idx += 1
        net.train()
        input_image = input_image[0].to(device)
        optimizer_G.zero_grad()
        aux_optimizer.zero_grad()
        global_step += 1
        start_time = time.time()

        results = net(input_image)
        out_criterion = criterion(results, input_image)
        out_criterion["loss"].backward()

        if config.clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), config.clip_max_norm)
        optimizer_G.step()

        aux_loss = net.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        elapsed.update(time.time() - start_time)

        losses.update(out_criterion["loss"].item())
        bppys.update(out_criterion["bpp_y"].item())
        bppzs.update(out_criterion["bpp_z"].item())

        psnr_ntc = 10 * (torch.log(1. / out_criterion["mse_loss_ntc"]) / np.log(10))
        psnr_ntcs.update(psnr_ntc.item())

        losses.update(out_criterion["loss"].item())

        if (global_step % config.print_every) == 0:
            process = (global_step % train_loader.__len__()) / (train_loader.__len__()) * 100.0
            log_info = [
                f'Step [{global_step % train_loader.__len__()}/{train_loader.__len__()}={process:.2f}%]',
                f'Loss {losses.val:.3f} ({losses.avg:.3f})',
                f'Time {elapsed.avg:.2f}',
                f'PSNR {psnr_ntcs.val:.2f} ({psnr_ntcs.avg:.2f})',
                f'Bpp_y {bppys.val:.2f} ({bppys.avg:.2f})',
                f'Bpp_z {bppzs.val:.4f} ({bppzs.avg:.4f})',
                f'Epoch {epoch}'
            ]
            log = (' | '.join(log_info))
            logger.info(log)
            if config.wandb:
                log_dict = {"Bpp_y": bppys.avg,
                            "Bpp_z": bppzs.avg, "loss": losses.avg, "Step": global_step // config.print_every,
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
                        "aux_optimizer": aux_optimizer.state_dict(),
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
                    "aux_optimizer": aux_optimizer.state_dict(),
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
        elapsed, losses, bst_losses, bppys, bppzs, psnr_ntcs = [AverageMeter() for _ in range(6)]
        print(len(test_loader))
        batch_idx = 0
        for input_image in test_loader:
            start_time = time.time()
            input_image = input_image[0].to(device)
            print(input_image.shape)
            num_pixels = input_image.shape[0] * input_image.shape[2] * input_image.shape[3]
            print(num_pixels)
            if config.test_entropy_estimation:
                output = net(input_image)
                y_likelihoods = output["likelihoods"]['y']
                z_likelihoods = output["likelihoods"]['z']
                bpp_y = torch.log(y_likelihoods).sum() / (-math.log(2) * num_pixels)
                bpp_z = torch.log(z_likelihoods).sum() / (-math.log(2) * num_pixels)
                mse = criterion.mse(output["x_hat"], input_image)
                loss = 0.01 * 255 ** 2 * mse + config.train_lambda * (bpp_y + bpp_z)
            else:
                out_enc = net.compress(input_image)
                y_string = out_enc["strings"][0]
                z_string = out_enc["strings"][1]
                string_recon = [y_string, z_string]
                shape = out_enc["shape"]

                out_dec = net.decompress(string_recon, shape)
                bpp_y = sum(len(s) for s in y_string) * 8.0 / num_pixels
                bpp_z = sum(len(s) for s in z_string) * 8.0 / num_pixels

                mse = criterion.mse(out_dec["x_hat"], input_image)
                loss = 0.01 * 255 ** 2 * mse + config.train_lambda * (bpp_y + bpp_z)
            elapsed.update(time.time() - start_time)
            losses.update(loss.item())
            bppys.update(bpp_y)
            bppzs.update(bpp_z)

            psnr_ntc = 10 * (torch.log(1. / mse) / np.log(10))
            psnr_ntcs.update(psnr_ntc.item())
            log = (' | '.join([
                f'Step [{(batch_idx + 1)}/{test_loader.__len__()}]',
                f'Loss {losses.val:.3f} ({losses.avg:.3f})',
                f'Time {elapsed.avg:.2f}',
                f'PSNR {psnr_ntcs.val:.2f} ({psnr_ntcs.avg:.2f})',
                f'Bpp_y {bppys.val:.2f} ({bppys.avg:.2f})',
                f'Bpp_z {bppzs.val:.4f} ({bppzs.avg:.4f})'
            ]))
            logger.info(log)
            batch_idx += 1
    if not config.test_only and config.wandb:
        wandb.log({"[Kodak] PSNR_NTC": psnr_ntcs.avg,
                   "[Kodak] Bpp_y": bppys.avg,
                   "[Kodak] Bpp_z": bppzs.avg,
                   "[Kodak] loss": bst_losses.avg})
    return bst_losses.avg


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
    #train_loader, test_loader = get_cifar10_loader(config)
    train_loader, test_loader = get_loader(config)
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
        aux_optimizer = optim.Adam(aux_params, lr=config.aux_lr)
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

        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer_G, milestones=[1000000], gamma=0.1)

        steps_epoch = global_step // train_loader.__len__()
        for epoch in range(steps_epoch, tot_epoch):
            logger.info('======Current epoch %s ======' % epoch)
            logger.info(f"Learning rate: {optimizer_G.param_groups[0]['lr']}")
            train_one_epoch(epoch, net, criterion, train_loader, test_loader, optimizer_G, aux_optimizer,
                            lr_scheduler, device, logger)


if __name__ == '__main__':
    main(sys.argv[1:])
