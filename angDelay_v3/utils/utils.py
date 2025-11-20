import numpy as np
import matplotlib.pyplot as plt
import math
import torch
import torchvision
import random
import os
from torch.autograd import Variable
import logging
from PIL import Image
import datetime
import time
#from csi_newFramework.angDelay.loss.distortion import MS_SSIM
from scipy import signal
from scipy import ndimage


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


def count_params(model):
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print("TOTAL Params {}M".format(num_params / 10 ** 6))


def BLN2BCHW(x, H, W):
    B, L, N = x.shape
    return x.reshape(B, H, W, N).permute(0, 3, 1, 2)


def BCHW2BLN(x):
    return x.flatten(2).permute(0, 2, 1)


def CalcuPSNR(img1, img2, normalize=False, max_val=255.):
    """
    Based on `tf.image.psnr`
    https://www.tensorflow.org/api_docs/python/tf/image/psnr
    """
    float_type = 'float64'
    if normalize:
        img1 = (torch.clamp(img1, -1, 1).cpu().numpy() + 1) / 2 * 255
        img2 = (torch.clamp(img2, -1, 1).cpu().numpy() + 1) / 2 * 255
    else:
        img1 = torch.clamp(img1, 0, 1).cpu().numpy() * 255
        img2 = torch.clamp(img2, 0, 1).cpu().numpy() * 255

    img1 = img1.astype(float_type)
    img2 = img2.astype(float_type)
    mse = np.mean(np.square(img1 - img2), axis=(1, 2, 3))
    psnr = 20 * np.log10(max_val) - 10 * np.log10(mse)
    return psnr


def cal_psnr(ref, target):
    diff = ref / 255.0 - target / 255.0
    diff = diff.flatten('C')
    rmse = math.sqrt(np.mean(diff ** 2.))
    return 20 * math.log10(1.0 / (rmse))


def fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x, y = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    return g / g.sum()


def ssim(img1, img2, cs_map=False):
    """Return the Structural Similarity Map corresponding to input images img1
    and img2 (images are assumed to be uint8)

    This function attempts to mimic precisely the functionality of ssim.m a
    MATLAB provided by the author's of SSIM
    https://ece.uwaterloo.ca/~z70wang/research/ssim/ssim_index.m
    """
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    size = 11
    sigma = 1.5
    window = fspecial_gauss(size, sigma)
    K1 = 0.01
    K2 = 0.03
    L = 255  # bitdepth of image
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    mu1 = signal.fftconvolve(window, img1, mode='valid')
    mu2 = signal.fftconvolve(window, img2, mode='valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = signal.fftconvolve(window, img1 * img1, mode='valid') - mu1_sq
    sigma2_sq = signal.fftconvolve(window, img2 * img2, mode='valid') - mu2_sq
    sigma12 = signal.fftconvolve(window, img1 * img2, mode='valid') - mu1_mu2
    if cs_map:
        return (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)),
                (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))
    else:
        return ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))


def cal_msssim(img1, img2):
    """This function implements Multi-Scale Structural Similarity (MSSSIM) Image
    Quality Assessment according to Z. Wang's "Multi-scale structural similarity
    for image quality assessment" Invited Paper, IEEE Asilomar Conference on
    Signals, Systems and Computers, Nov. 2003

    Author's MATLAB implementation:-
    http://www.cns.nyu.edu/~lcv/ssim/msssim.zip
    """
    level = 5
    weight = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    downsample_filter = np.ones((2, 2)) / 4.0
    im1 = img1.astype(np.float64)
    im2 = img2.astype(np.float64)
    mssim = np.array([])
    mcs = np.array([])
    for l in range(level):
        ssim_map, cs_map = ssim(im1, im2, cs_map=True)
        mssim = np.append(mssim, ssim_map.mean())
        mcs = np.append(mcs, cs_map.mean())
        filtered_im1 = ndimage.filters.convolve(im1, downsample_filter, mode='reflect')
        filtered_im2 = ndimage.filters.convolve(im2, downsample_filter, mode='reflect')
        im1 = filtered_im1[::2, ::2]
        im2 = filtered_im2[::2, ::2]
    return (np.prod(mcs[0:level - 1] ** weight[0:level - 1]) * (mssim[level - 1] ** weight[level - 1]))


def MSE2PSNR(MSE):
    return 10 * math.log10(255 ** 2 / (MSE))


def np2tensor(np_obj):
    # change dimenion of np array into tensor array
    return torch.Tensor(np_obj[:, :, :, np.newaxis].transpose((3, 2, 0, 1)))


def logger_configuration(config, log_name, save_log=False, test_mode=False):
    # 配置 logger
    logger = logging.getLogger("Deep joint source channel coder %s" % log_name)
    if test_mode:
        config.workdir += '_test'
    if save_log:
        makedirs(config.workdir)
        makedirs(config.samples)
        makedirs(config.models)
    formatter = logging.Formatter("%(asctime)s;%(levelname)s;%(message)s",
                                  "%Y-%m-%d %H:%M:%S")
    stdhandler = logging.StreamHandler()
    stdhandler.setLevel(logging.INFO)
    stdhandler.setFormatter(formatter)
    logger.addHandler(stdhandler)
    if save_log:
        filehandler = logging.FileHandler(log_name)
        filehandler.setLevel(logging.INFO)
        filehandler.setFormatter(formatter)
        logger.addHandler(filehandler)
    logger.setLevel(logging.INFO)
    config.logger = logger
    return config.logger


def Var(x, device):
    return Variable(x.to(device))


def single_plot(epoch, global_step, real, gen, config, normalize=True, single_compress=False):
    # print(real.shape)
    # real = real.permute(1, 2, 0)
    # gen = gen.permute(1, 2, 0)
    images = [real, gen]

    # comparison = np.hstack(images)

    # old save_fig

    # f = plt.figure()
    # plt.imshow(comparison)
    # plt.axis('off')
    # if single_compress:
    #     f.savefig(config.name, format='png', dpi=720, bbox_inches='tight', pad_inches=0)
    # else:
    #     f.savefig("{}/JSCCModel_{}_epoch{}_step{}.png".format(config.samples, config.trainset, epoch, global_step),
    #               format='png', dpi=720, bbox_inches='tight', pad_inches=0)
    # plt.gcf().clear()
    # plt.close(f)

    # new save_fig
    filename = "{}/JSCCModel_{}_epoch{}_step{}.png".format(config.samples, config.trainset, epoch, global_step)
    torchvision.utils.save_image(images, filename)


def triple_plot(filename, real, gen1, gen2):
    images = torch.cat((real, gen1, gen2), dim=0)
    torchvision.utils.save_image(images, filename)


def single_plot_II(epoch, global_step, real, gen_I, gen_II, config, single_compress=False):
    real = real.transpose([1, 2, 0])
    gen_I = gen_I.transpose([1, 2, 0])
    gen_II = gen_II.transpose([1, 2, 0])
    images = list()

    for im, imtype in zip([real, gen_I, gen_II], ['real', 'gen_I', 'gen_II']):
        im = ((im + 1.0)) / 2  # [-1,1] -> [0,1]
        im = np.squeeze(im)
        if len(im.shape) == 3:
            im = im[:, :, :3]
        if len(im.shape) == 4:
            im = im[0, :, :, :3]
        images.append(im)

    comparison = np.hstack(images)
    f = plt.figure()
    plt.imshow(comparison)
    plt.axis('off')
    if single_compress:
        f.savefig(config.name, format='png', dpi=720, bbox_inches='tight', pad_inches=0)
    else:
        f.savefig("{}/gan_compression_{}_epoch{}_step{}_{}_comparison.png".format(config.samples, config.name, epoch,
                                                                                  global_step, imtype),
                  format='png', dpi=720, bbox_inches='tight', pad_inches=0)
    plt.gcf().clear()
    plt.close(f)


def makedirs(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def load_model(model, f):
    with open(f, 'rb') as f:
        pretrained_dict = torch.load(f)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    f = str(f)
    if f.find('iter') != -1 and f.find('.model') != -1:
        st = f.find('iter') + 4
        ed = f.find('.model', st)
        return int(f[st:ed])
    else:
        return 0


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def adjust_learning_rate(optimizer, base_lr, global_step, warmup_step, decay_interval, lr_decay):
    global cur_lr
    if global_step < warmup_step:
        lr = base_lr * global_step / warmup_step
    elif global_step < decay_interval:
        lr = base_lr
    else:
        lr = base_lr * lr_decay
    cur_lr = lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_learning_rate_to(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def read_frame_to_torch(path):
    input_image = Image.open(path).convert('RGB')
    input_image = np.asarray(input_image).astype('float64').transpose(2, 0, 1)
    input_image = torch.from_numpy(input_image).type(torch.FloatTensor)
    input_image = input_image.unsqueeze(0) / 255
    return input_image


def write_torch_frame(frame, path):
    frame_result = frame.clone()
    frame_result = frame_result.cpu().detach().numpy().transpose(1, 2, 0) * 255
    frame_result = np.clip(np.rint(frame_result), 0, 255)
    frame_result = Image.fromarray(frame_result.astype('uint8'), 'RGB')
    frame_result.save(path)


def bpp_snr_to_kdivn(bpp, SNR, Nt):
    snr = 10 ** (SNR / 10)
    # kdivn = bpp / 3 / np.log2(1 + snr)
    kdivn = bpp / 2 / np.log2(1 + Nt * snr)
    return kdivn
