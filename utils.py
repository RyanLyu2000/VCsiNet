import logging
import os
import matplotlib.pyplot as plt
import torch
import math
import numpy as np
def logger_configuration(filename, phase, method='NTC', save_log=True):
    logger = logging.getLogger("NTSCC")
    workdir = './history/{}/{}'.format(method, filename)
    if phase == 'test':
        workdir += '_test'
    log = workdir + '/{}.log'.format(filename)
    samples = workdir + '/samples'
    models = workdir + '/models'
    if save_log:
        makedirs(workdir)
        makedirs(samples)
        makedirs(models)

    formatter = logging.Formatter("%(asctime)s;%(levelname)s;%(message)s",
                                  "%Y-%m-%d %H:%M:%S")
    stdhandler = logging.StreamHandler()
    stdhandler.setLevel(logging.INFO)
    stdhandler.setFormatter(formatter)
    logger.addHandler(stdhandler)
    if save_log:
        filehandler = logging.FileHandler(log)
        filehandler.setLevel(logging.INFO)
        filehandler.setFormatter(formatter)
        logger.addHandler(filehandler)
    logger.setLevel(logging.INFO)
    return workdir, logger


def makedirs(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_checkpoint(state, is_best, base_dir, filename="checkpoint.pth.tar"):
    if is_best:
        torch.save(state, base_dir + "/checkpoint_best_loss.pth.tar")
    else:
        torch.save(state, base_dir + "/" + filename)


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def clear(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0


def load_weights(net, model_path, load_featenc=True):
    try:
        pretrained = torch.load(model_path)['state_dict']
    except:
        pretrained = torch.load(model_path)
    result_dict = {}
    for key, weight in pretrained.items():
        # for wsx ntc pretrained model
        # if key[4:7] == 'g_a' or key[4:7] == 'h_a' or key[4:7] == 'g_s' or key[4:7] == 'h_s':
        #     result_key = key[4:7] + '.' + key[4:]
        # else:
        #     result_key = key[4:]

        # for wsx ntscc pretrained model
        # if key[4:7] == 'g_a' or key[4:7] == 'h_a' or key[4:7] == 'g_s' or key[4:7] == 'h_s':
        #     result_key = key[0:4] + key[4:7] + '.' + key[4:]
        # else:
        #     result_key = key

        result_key = key
        if load_featenc:
            if 'mask' not in key:
                result_dict[result_key] = weight
        else:
            # if 'attn_mask' not in key and 'rate_adaption.mask' not in key\
            #         and 'fe' not in key and 'fd' not in key:
            result_dict[result_key] = weight
    print(net.load_state_dict(result_dict, strict=False))
    del result_dict, pretrained


def load_checkpoint(logger, device, global_step, net, optimizer_G, aux_optimizer, model_path):
    logger.info("Loading " + str(model_path))
    pre_dict = torch.load(model_path, map_location=device)

    global_step = pre_dict["global_step"]

    result_dict = {}
    for key, weight in pre_dict["state_dict"].items():
        result_key = key
        if 'mask' not in key:
            result_dict[result_key] = weight
    net.load_state_dict(result_dict, strict=False)

    # optimizer_G.load_state_dict(pre_dict["optimizer"])
    aux_optimizer.load_state_dict(pre_dict["aux_optimizer"])
    # lr_scheduler.load_state_dict(pre_dict["lr_scheduler"])

    return global_step

# def sample_beta(n, config):
#     low, high = config.train_beta_range  # original beta space
#     p = 3.0
#     low, high = math.pow(low, 1 / p), math.pow(high, 1 / p)  # transformed space
#     transformed_lmb = low + (high - low) * torch.rand(1, device=config.device)
#     lmb = torch.pow(transformed_lmb, exponent=p)
#     return lmb

def sample_beta(n, config):
    low, high = config.train_beta_range  # original beta space
    transformed_lmb = low + (high - low) * torch.rand(1, device=config.device)
    return transformed_lmb

def cal_cov(mat):
    cov_list = []
    for i in range(mat.shape[0]):
        cov_list.append(np.cov(mat[i], rowvar=False))
    return np.array(cov_list)

def plot_heatmap(map1, map2, map3, path):
    plt.rcParams['figure.figsize'] = (14, 4)
    plt.subplot(1, 3, 1)
    plt.imshow(map1)
    plt.colorbar(shrink=0.5)
    plt.title("y_cov")
    plt.subplot(1, 3, 2)
    plt.imshow(map2)
    plt.colorbar(shrink=0.5)
    plt.title("w_cov")

    plt.subplot(1, 3, 3)
    plt.imshow(map3)
    plt.colorbar(shrink=0.5)
    plt.title("y_cov/w_cov")
    plt.savefig(path, dpi=200, bbox_inches='tight')

def plot_mean_std(mean1, std1, mean2, std2, mean3, std3, path):
    plt.rcParams['figure.figsize'] = (9, 4)
    x = np.arange(0, 64, 1)
    plt.subplot(1, 2, 1)
    a, = plt.plot(x, mean1, "-s", color="r", linewidth=1, markersize=2, label='mean of w (learned)')
    b, = plt.plot(x, mean2, "-s", color="b", linewidth=1, markersize=2, label='mean of w (from statistics)')
    c, = plt.plot(x, mean3, "-s", color="g", linewidth=1, markersize=2, label='mean of y (from statistics)')
    plt.legend(handles=[a, b, c])
    plt.title("mean")

    plt.subplot(1, 2, 2)
    a, = plt.plot(x, std1, "-s", color="r", linewidth=1, markersize=2, label='std of w (learned)')
    b, = plt.plot(x, std2, "-s", color="b", linewidth=1, markersize=2, label='std of w (from statistics)')
    c, = plt.plot(x, std3, "-s", color="g", linewidth=1, markersize=2, label='std of y (from statistics)')
    plt.legend(handles=[a, b, c])
    plt.title("std")
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()

def plot_entropy(e1, e2, e3, path):
    plt.rcParams['figure.figsize'] = (6, 4)
    x = np.arange(0, e1.shape[0], 1)
    a, = plt.plot(x, e1, "-s", color="r", linewidth=1, markersize=2, label='entropy of y')
    b, = plt.plot(x, e2, "-s", color="b", linewidth=1, markersize=2, label='entropy of w')
    c, = plt.plot(x, e3, "-s", color="g", linewidth=1, markersize=2, label='entropy of w(learned)')
    plt.legend(handles=[a, b, c])
    plt.xlabel('channels')
    plt.ylabel('entropy')
    plt.title("entropy of of w&y")
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()