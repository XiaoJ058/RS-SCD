import os
import time
import argparse
import numpy as np
import torch
import torch.autograd
from skimage import io, exposure
from torch.nn import functional as F
from torch.utils.data import DataLoader

#################################
from datasets import RS_ST as RS
from models.Compar.DFINet import DFINet as Net
DATA_NAME = 'ST'
#################################


class PredOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        working_path = os.path.dirname(os.path.abspath(__file__))
        parser.add_argument('--pred_batch_size', default=1, help='prediction batch size')
        parser.add_argument('--test_dir', default=R"I:\Datasets\SCD\SECOND_OG\test", help='directory to test images')
        parser.add_argument('--pred_dir', default=R"I:\Datasets\SCD\SECOND_OG\DFINet", help='directory to output masks')
        parser.add_argument('--chkpt_path', default=working_path + R'\checkpoints/Compar_Method_SECOND/DFINet_43e_mIoU72.61_Sek31.38_OA89.11.pth')
        parser.add_argument('--no-tta', action='store_true', help='Disable test time augmentation (enabled by default)')
        self.initialized = True
        return parser

    def gather_options(self):
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        self.parser = parser
        return parser.parse_args()

    def parse(self):
        self.opt = self.gather_options()
        return self.opt


def get_tta_transforms(img_A, img_B):
    """
    Return a list of (transformed_A, transformed_B, inverse_transform)
    for test time augmentation. Implements:
        - identity
        - vertical flip
        - horizontal flip
        - diagonal flip (vertical + horizontal)
        - rotation 90°
        - rotation 180°
        - rotation 270°
    """
    transforms = []
    # Identity
    transforms.append((img_A, img_B, lambda x: x))
    # Vertical flip
    transforms.append((torch.flip(img_A, [2]), torch.flip(img_B, [2]), lambda x: torch.flip(x, [2])))
    # Horizontal flip
    transforms.append((torch.flip(img_A, [3]), torch.flip(img_B, [3]), lambda x: torch.flip(x, [3])))
    # Diagonal flip (vertical + horizontal)
    transforms.append((torch.flip(img_A, [2, 3]), torch.flip(img_B, [2, 3]), lambda x: torch.flip(x, [2, 3])))
    # Rotation 90°
    transforms.append((torch.rot90(img_A, k=1, dims=[2,3]), torch.rot90(img_B, k=1, dims=[2,3]),
                       lambda x: torch.rot90(x, k=-1, dims=[2,3])))
    # Rotation 180°
    transforms.append((torch.rot90(img_A, k=2, dims=[2,3]), torch.rot90(img_B, k=2, dims=[2,3]),
                       lambda x: torch.rot90(x, k=2, dims=[2,3])))
    # Rotation 270°
    transforms.append((torch.rot90(img_A, k=3, dims=[2,3]), torch.rot90(img_B, k=3, dims=[2,3]),
                       lambda x: torch.rot90(x, k=1, dims=[2,3])))
    return transforms


def predict(net, pred_set, pred_loader, pred_dir, tta=False, index_map=False, intermediate=False):
    """
    For models with 3 outputs: change map + 2 semantic maps.
    tta: if True, apply test time augmentation (flip + rotation ensemble)
    index_map: if True, save class index maps (values 0-3)
    intermediate: if True, save intermediate semantic maps and change map before masking
    """
    pred_A_dir_rgb = os.path.join(pred_dir, 'im1_rgb')
    pred_B_dir_rgb = os.path.join(pred_dir, 'im2_rgb')
    os.makedirs(pred_A_dir_rgb, exist_ok=True)
    os.makedirs(pred_B_dir_rgb, exist_ok=True)

    if index_map:
        pred_A_dir = os.path.join(pred_dir, 'im1')
        pred_B_dir = os.path.join(pred_dir, 'im2')
        os.makedirs(pred_A_dir, exist_ok=True)
        os.makedirs(pred_B_dir, exist_ok=True)

    if intermediate:
        pred_mA_dir = os.path.join(pred_dir, 'im1_semantic')
        pred_mB_dir = os.path.join(pred_dir, 'im2_semantic')
        pred_change_dir = os.path.join(pred_dir, 'change')
        os.makedirs(pred_mA_dir, exist_ok=True)
        os.makedirs(pred_mB_dir, exist_ok=True)
        os.makedirs(pred_change_dir, exist_ok=True)

    for vi, data in enumerate(pred_loader):
        imgs_A, imgs_B = data
        imgs_A = imgs_A.cuda().float()
        imgs_B = imgs_B.cuda().float()
        mask_name = pred_set.get_mask_name(vi)

        with torch.no_grad():
            if tta:
                change_sum = 0
                semA_sum = 0
                semB_sum = 0
                n_tta = 0
                for tA, tB, inv in get_tta_transforms(imgs_A, imgs_B):
                    out_c, out_A, out_B = net(tA, tB)
                    out_c = inv(out_c)
                    out_A = inv(out_A)
                    out_B = inv(out_B)
                    change_sum += F.sigmoid(out_c)
                    semA_sum += F.softmax(out_A, dim=1)
                    semB_sum += F.softmax(out_B, dim=1)
                    n_tta += 1
                change_logit = change_sum / n_tta
                semA = semA_sum / n_tta
                semB = semB_sum / n_tta
            else:
                out_c, out_A, out_B = net(imgs_A, imgs_B)
                change_logit = F.sigmoid(out_c)
                semA = F.softmax(out_A, dim=1)
                semB = F.softmax(out_B, dim=1)

        semA = semA.cpu()
        semB = semB.cpu()
        change_mask = (change_logit.cpu() > 0.5).squeeze().numpy()

        pred_A = torch.argmax(semA, dim=1).squeeze().numpy()
        pred_B = torch.argmax(semB, dim=1).squeeze().numpy()

        if intermediate:
            io.imsave(os.path.join(pred_mA_dir, mask_name), RS.Index2Color(pred_A))
            io.imsave(os.path.join(pred_mB_dir, mask_name), RS.Index2Color(pred_B))
            change_map_img = exposure.rescale_intensity(change_mask, 'image', 'dtype')
            io.imsave(os.path.join(pred_change_dir, mask_name), change_map_img)

        pred_A_masked = (pred_A * change_mask).astype(np.uint8)
        pred_B_masked = (pred_B * change_mask).astype(np.uint8)

        io.imsave(os.path.join(pred_A_dir_rgb, mask_name), RS.Index2Color(pred_A_masked))
        io.imsave(os.path.join(pred_B_dir_rgb, mask_name), RS.Index2Color(pred_B_masked))
        print(os.path.join(pred_A_dir_rgb, mask_name))

        if index_map:
            io.imsave(os.path.join(pred_A_dir, mask_name), (pred_A_masked * 255).astype(np.uint8))
            io.imsave(os.path.join(pred_B_dir, mask_name), (pred_B_masked * 255).astype(np.uint8))


def main():
    begin_time = time.time()
    opt = PredOptions().parse()
    net = Net(num_classes=RS.num_classes).cuda()
    net.load_state_dict(torch.load(opt.chkpt_path))
    net.eval()

    test_set = RS.Data_test(opt.test_dir)
    test_loader = DataLoader(test_set, batch_size=opt.pred_batch_size)

    # TTA 默认启用，可通过 --no-tta 关闭
    tta_enabled = not opt.no_tta
    predict(net, test_set, test_loader, opt.pred_dir,
            tta=tta_enabled, index_map=True, intermediate=True)

    time_use = time.time() - begin_time
    print('Total time: %.2fs' % time_use)


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    main()