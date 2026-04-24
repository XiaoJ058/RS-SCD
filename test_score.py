import os
import time
import argparse
import torch.autograd
from torch.nn import functional as F
from torch.utils.data import DataLoader
from datasets import RS_ST as RS
from models.Compar.DFINet import DFINet as Net
from utils.utils import accuracy, SCDD_eval_all, AverageMeter

DATA_NAME = 'ST'


class PredOptions():
    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        working_path = os.path.dirname(os.path.abspath(__file__))
        parser.add_argument('--pred_batch_size', required=False, default=8, help='prediction batch size')
        parser.add_argument('--test_dir', required=False, default=R"I:\Datasets\SCD\SECOND_OG\test",
                            help='directory to test images')
        parser.add_argument('--chkpt_path', required=False,
                            default=working_path + R'\checkpoints/Compar_Method_SECOND/DFINet_43e_mIoU72.61_Sek31.38_OA89.11.pth')
        self.initialized = True
        return parser

    def gather_options(self):
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        self.parser = parser
        return parser.parse_args()

    def parse(self):
        self.opt = self.gather_options()
        return self.opt


def main():
    begin_time = time.time()
    opt = PredOptions().parse()
    net = Net(num_classes=RS.num_classes).cuda()
    net.load_state_dict(torch.load(opt.chkpt_path))
    net.eval()

    test_set = RS.Data(opt.test_dir)
    test_loader = DataLoader(test_set, batch_size=opt.pred_batch_size)
    Fscd, mIoU_v, Sek, acc_v = test(test_loader, net)
    # predict(net, test_set, test_loader, opt.pred_dir, flip=False, index_map=True, intermediate=False)
    # predict_direct(net, test_set, test_loader, opt.pred_dir, flip=False, index_map=True)
    time_use = time.time() - begin_time
    print('Total time: %.2fs' % time_use)


def horizontal_flip(x):
    return torch.flip(x, dims=[3])


def vertical_flip(x):
    return torch.flip(x, dims=[2])


def test(val_loader, net):
    # the following code is written assuming that batch size is 1
    net.eval()
    torch.cuda.empty_cache()
    start = time.time()

    acc_meter = AverageMeter()

    preds_all = []
    labels_all = []
    for vi, data in enumerate(val_loader):
        imgs_A, imgs_B, labels_A, labels_B = data
        imgs_A = imgs_A.cuda().float()
        imgs_B = imgs_B.cuda().float()
        labels_A = labels_A.cuda().long()
        labels_B = labels_B.cuda().long()

        with torch.no_grad():
            preds_outchange2 = []
            preds_outputs_A = []
            preds_outputs_B = []

            out_change2, outputs_A, outputs_B = net(imgs_A, imgs_B)
            preds_outchange2.append(out_change2)
            preds_outputs_A.append(outputs_A)
            preds_outputs_B.append(outputs_B)

            # TTA 2: horizontal flip
            out_change2, outputs_A, outputs_B = net(horizontal_flip(imgs_A), horizontal_flip(imgs_B))
            out_change2 = horizontal_flip(out_change2)
            outputs_A = horizontal_flip(outputs_A)
            outputs_B = horizontal_flip(outputs_B)
            preds_outchange2.append(out_change2)
            preds_outputs_A.append(outputs_A)
            preds_outputs_B.append(outputs_B)

            # TTA 3: vertical flip
            out_change2, outputs_A, outputs_B = net(vertical_flip(imgs_A), vertical_flip(imgs_B))
            out_change2 = vertical_flip(out_change2)
            outputs_A = vertical_flip(outputs_A)
            outputs_B = vertical_flip(outputs_B)
            preds_outchange2.append(out_change2)
            preds_outputs_A.append(outputs_A)
            preds_outputs_B.append(outputs_B)

            # TTA 4: horizontal + vertical flip
            out_change2, outputs_A, outputs_B = net(vertical_flip(horizontal_flip(imgs_A)), vertical_flip(horizontal_flip(imgs_B)))
            out_change2 = vertical_flip(horizontal_flip(out_change2))
            outputs_A = vertical_flip(horizontal_flip(outputs_A))
            outputs_B = vertical_flip(horizontal_flip(outputs_B))
            preds_outchange2.append(out_change2)
            preds_outputs_A.append(outputs_A)
            preds_outputs_B.append(outputs_B)

            # 平均融合
            out_change2 = torch.stack(preds_outchange2).mean(dim=0)
            outputs_A = torch.stack(preds_outputs_A).mean(dim=0)
            outputs_B = torch.stack(preds_outputs_B).mean(dim=0)

        labels_A = labels_A.cpu().detach().numpy()
        labels_B = labels_B.cpu().detach().numpy()
        outputs_A = outputs_A.cpu().detach()
        outputs_B = outputs_B.cpu().detach()
        change_mask = F.sigmoid(out_change2).cpu().detach() > 0.5
        preds_A = torch.argmax(outputs_A, dim=1)
        preds_B = torch.argmax(outputs_B, dim=1)
        preds_A = (preds_A*change_mask.squeeze().long()).numpy()
        preds_B = (preds_B*change_mask.squeeze().long()).numpy()
        for (pred_A, pred_B, label_A, label_B) in zip(preds_A, preds_B, labels_A, labels_B):
            acc_A, valid_sum_A = accuracy(pred_A, label_A)
            acc_B, valid_sum_B = accuracy(pred_B, label_B)
            preds_all.append(pred_A)
            preds_all.append(pred_B)
            labels_all.append(label_A)
            labels_all.append(label_B)
            acc = (acc_A + acc_B)*0.5
            acc_meter.update(acc)

    Fscd, IoU_mean, SeK = SCDD_eval_all(preds_all, labels_all, RS.num_classes)

    curr_time = time.time() - start
    print('Fscd: %.2f mIoU: %.2f SeK: %.2f Accuracy: %.2f'\
    %(Fscd*100, IoU_mean*100, SeK*100, acc_meter.average()*100))

    return Fscd, IoU_mean, SeK, acc_meter.avg


if __name__ == '__main__':
    main()