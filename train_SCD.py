import os
import time
import random
import numpy as np
import torch.nn as nn
import torch.autograd
from skimage import io
#io 对图片进行输入和输出
from torch import optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

working_path = os.path.dirname(os.path.abspath(__file__))
#print(__file__)
#打印文件当前的位置
#working_path:'d:\\dl_file\\py_file\\train_code\\classic_model\\Bi-SRNet'

from utils.loss import CrossEntropyLoss2d, weighted_BCE_logits, ChangeSimilarity
from utils.utils import accuracy, SCDD_eval_all, AverageMeter

#Data and model choose
from datasets import RS_ST as RS
# from models.BiSRNet import BiSRNet as Net
from models.SSCDl import SSCDl as Net
# from models.SSCDl import SSCDl as Net

NET_NAME = 'SSCDl+SCLoss'
DATA_NAME = 'SSCDl_0.02_3E'

# Training options
###############################################
args = {
    'train_batch_size': 4,
    'val_batch_size': 4,
    'lr': 0.02,
    'epochs': 70,
    'gpu': True,
    'lr_decay_power': 1.5,
    'weight_decay': 3e-4,
    'momentum': 0.9,
    'print_freq': 100,
    'predict_step': 5,
    'pred_dir': os.path.join(working_path, 'results', DATA_NAME),
    'chkpt_dir': os.path.join(working_path, 'checkpoints', DATA_NAME),
    'log_dir': os.path.join(working_path, 'logs', DATA_NAME, NET_NAME),
    'load_path': os.path.join(working_path, 'checkpoints', DATA_NAME, 'pretrained.pth')
}
#将要用的参数放在一起，便于调用
###############################################
if not os.path.exists(args['log_dir']): os.makedirs(args['log_dir'])
if not os.path.exists(args['pred_dir']): os.makedirs(args['pred_dir'])
if not os.path.exists(args['chkpt_dir']): os.makedirs(args['chkpt_dir'])
writer = SummaryWriter(args['log_dir'])   # 记录loss等指标到tensorboard中

def main():
    net = Net().cuda() # net.load_state_dict(torch.load(args['load_path']), strict=False)
    train_set = RS.Data('train', random_flip=True)  # 包含两个字典（包括了A/B数据的具体路径，label：具体的张量数据）
    train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=0, shuffle=True) # 一批是4的话，一共有416张图片则需要104批→dataloader = 104
    val_set = RS.Data('val')
    val_loader = DataLoader(val_set, batch_size=args['val_batch_size'], num_workers=0, shuffle=False)

    criterion = CrossEntropyLoss2d(ignore_index=0).cuda()    # 多分类交叉熵
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args['lr'], weight_decay=args['weight_decay'], momentum=args['momentum'], nesterov=True)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95, last_epoch=-1)    # 1指的是step_size,表示过多少个epoch,学习率更新；gamma表示更新要乘的系数

    train(train_loader, net, criterion, optimizer, scheduler, val_loader)
    writer.close()
    print('Training finished.')


def train(train_loader, net, criterion, optimizer, scheduler, val_loader, times=None):
    bestaccT = 0
    bestmIoU = 0.0
    bestloss = 1.0
    begin_time = time.time()
    all_iters = float(len(train_loader)*args['epochs'])  # train_loader=num(images)/batch_size
    criterion_sc = ChangeSimilarity().cuda()             # LOSSsc 计算P1\P2余弦相似度结合change label
    curr_epoch = 0
    while True:
        torch.cuda.empty_cache()   # 释放显存
        net.train()                # 用于训练，启用dropout和bn；还有net.eval()
        # freeze_model(net.CotSR)
        # freeze_model(net.VSD)     # 冻结模块做消融实验
        # freeze_model(net.VTS1)
        # freeze_model(net.res2)
        start = time.time()
        acc_meter = AverageMeter()   # 采用平均数
        train_seg_loss = AverageMeter()
        train_bn_loss = AverageMeter()
        train_sc_loss = AverageMeter()

        curr_iter = curr_epoch*len(train_loader)


        for i, data in enumerate(train_loader):            # 在这里开始投入数据进行训练；分别获取序号和对应的数据
            running_iter = curr_iter+i+1
            adjust_lr(optimizer, running_iter, all_iters)
            imgs_A, imgs_B, labels_A, labels_B = data
            if args['gpu']:
                imgs_A = imgs_A.cuda().float()
                imgs_B = imgs_B.cuda().float()
                labels_bn = (labels_A > 0).unsqueeze(1).cuda().float()  # 匹配维度,才能进行cd,添加一个维度变成(4,1,512,512)
                labels_A = labels_A.cuda().long()
                labels_B = labels_B.cuda().long()

            optimizer.zero_grad()   # 梯度清零
            out_change, outputs_A, outputs_B = net(imgs_A, imgs_B)  # 产生变化检测结果、两个七分类的结果
            assert outputs_A.size()[1] == RS.num_classes   # Python assert（断言）用于判断一个表达式，在表达式条件为 false 的时候触发异常

            loss_seg = criterion(outputs_A, labels_A) * 0.5 + criterion(outputs_B, labels_B) * 0.5
            loss_bn = weighted_BCE_logits(out_change, labels_bn)
            loss_sc = criterion_sc(outputs_A[:,1:], outputs_B[:,1:], labels_bn)
            loss = loss_seg + loss_bn + loss_sc

            loss.backward()   # 损失的回传
            optimizer.step()  # 模型参数的更新

            labels_A = labels_A.cpu().detach().numpy()   # tensor转为numpy数据。
            labels_B = labels_B.cpu().detach().numpy()
            outputs_A = outputs_A.cpu().detach()
            outputs_B = outputs_B.cpu().detach()
            change_mask = F.sigmoid(out_change).cpu().detach()>0.5
            # 利用sigmoid激活函数来对变化检测结果进行二值图处理
            # 阈值为0.5
            # detach()就是阻断反向传播，这里只想看输出结果
            preds_A = torch.argmax(outputs_A, dim=1)  # 指定dim=1既在7那个分类维度上,找出每个像素在七个维度上的最大值并显示该维度序号。目的就是为了分类
            preds_B = torch.argmax(outputs_B, dim=1)
            preds_A = (preds_A*change_mask.squeeze().long()).numpy() # 将二值化的掩膜(0,1)乘以上述分类过的PA\PB,达到SCD的目的。
            preds_B = (preds_B*change_mask.squeeze().long()).numpy()
            # batch_valid_sum = 0
            acc_curr_meter = AverageMeter()
            for (pred_A, pred_B, label_A, label_B) in zip(preds_A, preds_B, labels_A, labels_B):
                # zip()将括号内对象一一配对成一组
                acc_A, valid_sum_A = accuracy(pred_A, label_A)
                acc_B, valid_sum_B = accuracy(pred_B, label_B)
                acc = (acc_A + acc_B)*0.5
                acc_curr_meter.update(acc)
            acc_meter.update(acc_curr_meter.avg)
            train_seg_loss.update(loss_seg.cpu().detach().numpy())
            train_bn_loss.update(loss_bn.cpu().detach().numpy())
            train_sc_loss.update(loss_sc.cpu().detach().numpy())

            curr_time = time.time() - start

            if (i + 1) % args['print_freq'] == 0:
                # 打印的频率,i是(样本数量/batchsize),
                print('[epoch %d] [iter %d / %d %.1fs] [lr %f] [train seg_loss %.4f bn_loss %.4f acc %.2f]' % (
                    curr_epoch, i + 1, len(train_loader), curr_time, optimizer.param_groups[0]['lr'],
                    train_seg_loss.val, train_bn_loss.val, acc_meter.val*100)) #sc_loss %.4f, train_sc_loss.val,
                writer.add_scalar('train seg_loss', train_seg_loss.val, running_iter)
                writer.add_scalar('train sc_loss', train_sc_loss.val, running_iter)
                writer.add_scalar('train accuracy', acc_meter.val, running_iter)
                writer.add_scalar('lr', optimizer.param_groups[0]['lr'], running_iter)

        Fscd, mIoU_v, Kappa, acc_v, loss_v = validate(val_loader, net, criterion, curr_epoch)
        if acc_meter.avg>bestaccT: bestaccT=acc_meter.avg
        if mIoU_v > bestmIoU:
            bestmIoU=mIoU_v
            bestaccV=acc_v
            bestKappa = Kappa
            bestloss=loss_v
            torch.save(net.state_dict(), os.path.join(args['chkpt_dir'], NET_NAME+'_%de_mIoU%.2f_Sek%.2f_OA%.2f.pth'\
                %(curr_epoch, mIoU_v*100, bestKappa*100, acc_v*100)) )
            # 保存模型参数，后面直接将参数加载进模型中
            # torch.save(net,'xx.pth')
        print('Total time: %.1fs Best rec: Train acc %.2f,Val mIoU %.2f Val Sek %.2f acc %.2f loss %.4f'
              %(time.time()-begin_time, bestaccT*100, bestmIoU*100, bestKappa*100, bestaccV*100, bestloss))
        curr_epoch += 1
        # scheduler.step()
        if curr_epoch >= args['epochs']:
            return


def validate(val_loader, net, criterion, curr_epoch):
    # the following code is written assuming that batch size is 1
    net.eval()
    torch.cuda.empty_cache()
    start = time.time()

    val_loss = AverageMeter()
    acc_meter = AverageMeter()

    preds_all = []
    labels_all = []
    for vi, data in enumerate(val_loader):
        imgs_A, imgs_B, labels_A, labels_B = data
        if args['gpu']:
            imgs_A = imgs_A.cuda().float()
            imgs_B = imgs_B.cuda().float()
            labels_A = labels_A.cuda().long()
            labels_B = labels_B.cuda().long()

        with torch.no_grad():
            out_change, outputs_A, outputs_B = net(imgs_A, imgs_B)
            loss_A = criterion(outputs_A, labels_A)
            loss_B = criterion(outputs_B, labels_B)
            loss = loss_A * 0.5 + loss_B * 0.5
        val_loss.update(loss.cpu().detach().numpy())

        labels_A = labels_A.cpu().detach().numpy()
        labels_B = labels_B.cpu().detach().numpy()
        outputs_A = outputs_A.cpu().detach()
        outputs_B = outputs_B.cpu().detach()
        change_mask = F.sigmoid(out_change).cpu().detach()>0.5
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

        if curr_epoch%args['predict_step']==0 and vi==0:
            pred_A_color = RS.Index2Color(preds_A[0])
            pred_B_color = RS.Index2Color(preds_B[0])
            io.imsave(os.path.join(args['pred_dir'], NET_NAME+'_A.png'), pred_A_color)
            io.imsave(os.path.join(args['pred_dir'], NET_NAME+'_B.png'), pred_B_color)
            print('Prediction saved!')

    Fscd, IoU_mean, Kappa = SCDD_eval_all(preds_all, labels_all, RS.num_classes)

    curr_time = time.time() - start
    print('%.1fs Val loss: %.2f Fscd: %.2f IoU: %.2f Kappa: %.2f Accuracy: %.2f'\
    %(curr_time, val_loss.average(), Fscd*100, IoU_mean*100, Kappa*100, acc_meter.average()*100))

    writer.add_scalar('val_loss', val_loss.average(), curr_epoch)
    writer.add_scalar('val_Fscd', Fscd, curr_epoch)
    writer.add_scalar('val_Accuracy', acc_meter.average(), curr_epoch)
    #writer.add_scalar('myscalar', value, iteration)
    #来记录一个标量值；若X是张量，则用X.item()提取出标量

    return Fscd, IoU_mean, Kappa, acc_meter.avg, val_loss.avg


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False #将这个模块的参数梯度设为0，则这个模块就没啥用了
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()           #eval()一般用在测试中,不使用BN和dropout


def adjust_lr(optimizer, curr_iter, all_iter, init_lr=args['lr']):
    scale_running_lr = ((1. - float(curr_iter) / all_iter) ** args['lr_decay_power'])
    running_lr = init_lr * scale_running_lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = running_lr


if __name__ == '__main__':
    main()

