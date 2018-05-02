import argparse
import os,sys
import time
import shutil
import torch
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm
from log import log
from dataset import ClipDataset
#from p3d_model import *
#from p3d_model import get_optim_policies
#from I3D_Pytorch import *
from S3DG_Pytorch import *
from transforms import *
from IPython import embed
from dataloader_pkl import KineticsPKL
best_prec1=0
channel_dim=1

def get_args():
    parser = argparse.ArgumentParser(description="TAL")
    parser.add_argument('dataset', type=str, choices=['kinetics'])
    parser.add_argument('modality', type=str, choices=['RGB', 'Flow'])
    parser.add_argument('train_list', type=str)
    parser.add_argument('val_list', type=str)
    # ========================= Model Configs ==========================
    parser.add_argument('--arch', type=str, default="S3DG",choices=['S3DG'])
    parser.add_argument('--dropout', '--do', default=0.5, type=float, metavar='DO', help='dropout ratio (default: 0.5)')
    parser.add_argument('-d','--data_workers',default=8,type=int)
    # ========================= Learning Configs ==========================
    parser.add_argument('--epochs', default=45, type=int, metavar='N',help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=64, type=int,metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,metavar='LR', help='initial learning rate')
    parser.add_argument('--lr_steps', default=[20, 40], type=float, nargs="+",metavar='LRSteps', help='epochs to decay learning rate by 10')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('--clip-gradient', '--gd', default=None, type=float,metavar='W', help='gradient norm clipping (default: disabled)')
    parser.add_argument('--no_partialbn', '--npb', default=False, action="store_true")
    # ========================= Monitor Configs ==========================
    parser.add_argument('-p','--print-freq', default=20, type=int,metavar='N', help='print frequency (default: 10)')
    parser.add_argument('-ef','--eval-freq', default=5, type=int,metavar='N', help='evaluation frequency (default: 5)')
    # ========================= Runtime Configs ==========================
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',help='number of data loading workers (default: 4)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',help='evaluate model on validation set')
    parser.add_argument('--snapshot_pref', type=str, default="Qijie")
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',help='manual epoch number (useful on restarts)')
    parser.add_argument('--gpus', nargs='+', type=int, default=None)
    parser.add_argument('--flow_prefix', default="", type=str)
    # ========================= Return the final total Args==================
    Args=parser.parse_args()
    return Args


def main():
    global args,best_prec1
    args=get_args()

    log.l.info('Input command:\npython '+ ' '.join(sys.argv)+'  ===========>')
    
    if args.dataset == 'kinetics':
        num_class = 400
    else:
        raise ValueError('Unknown dataset '+args.dataset)

    log.l.info('============= prepare the model and model\'s parameters =============')

    if args.arch=='S3DG':
        input_channel=3 if args.modality=='RGB' else 2
        model=S3DG(num_classes=num_class,input_channel=input_channel,dropout_keep_prob=0.5)
        model.load_state_dict('modelweights/{}_imagenet.pkl'.format(args.modality))
    else:
        raise ValueError('Unknown model'+ args.arch)

    #model=transfer_model(model,num_classes=num_class)

    crop_size = 224# model.crop_size
    #scale_size = model.scale_size
    input_mean = [0.485,0.456,0.406]#model.input_mean
    input_std = [0.229,0.224,0.225]#model.input_std
    temporal_length = 64#model.temporal_length
   

    #model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()
    model=torch.nn.DataParallel(model).cuda()
    
    if args.resume:
        log.l.info('============== train from checkpoint (finetune mode) =================')
        if os.path.isfile(args.resume):
            log.l.info(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            log.l.info(("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch'])))
        else:
            log.l.info(("=> no checkpoint found at '{}'".format(args.resume)))


    log.l.info('============== Now, loading data ... ==============\n')
    # Data loading code
    normalize = GroupNormalize(input_mean, input_std)
    if args.modality == 'RGB':
        data_channel = 3
    elif args.modality == 'Flow':
        data_channel = 2 
    
    is_gray=False if args.modality=='RGB' else True
    Kinetics_train=KineticsPKL(args.train_list,seglen=64,is_train=True,cropsize=crop_size,
                   transform=torchvision.transforms.Compose([
                       GroupScale((256,256)),
                       #train_augmentation,
                       #GroupRandomCrop(crop_size),
                       GroupMultiScaleCrop(224,[1,0.875,0.75,0.66]),
                       GroupRandomHorizontalFlip(),
                       Stack(),
                       ToTorchFormatTensor(div=args.arch != 'BNInception'),
                       normalize,
                   ]))
    train_loader = torch.utils.data.DataLoader(
        Kinetics_train,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.data_workers, pin_memory=True,drop_last=True)

    Kinetics_val=KineticsPKL(args.val_list,seglen=64,is_train=False,cropsize=crop_size,
                   transform=torchvision.transforms.Compose([
                       GroupScale((256,256)),
                       GroupCenterCrop(224),
                       Stack(),
                       ToTorchFormatTensor(div=args.arch != 'BNInception'),
                       normalize,
                   ]))
    val_loader = torch.utils.data.DataLoader(
        Kinetics_val,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.data_workers, pin_memory=True,drop_last=True)
    log.l.info('================= Now, define loss function and optimizer ==============')
    
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1).cuda()
    
    #for group in policies:
    #    log.l.info(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
    #        group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))

    optimizer = torch.optim.SGD(model.parameters(),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    if args.evaluate:
        log.l.info('Need val the data first...')
        validate(val_loader, model, criterion, 0)

    log.l.info('\n\n===================> TRAIN and VAL begins <===================\n')

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr_steps)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            prec1 = validate(val_loader, model, criterion, (epoch + 1) * len(train_loader))

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best)

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()

    for i, (input, target,vid) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        # compute output
        output = model(input_var)[0]
        loss = criterion(output, target_var)
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1,5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))


        # compute gradient and do SGD step
        optimizer.zero_grad()

        loss.backward()

        if args.clip_gradient is not None:
            total_norm = clip_grad_norm(model.parameters(), args.clip_gradient)
            if total_norm > args.clip_gradient:
                log.l.info("clipping gradient: {} with coef {}".format(total_norm, args.clip_gradient / total_norm))

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            log.l.info(('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr'])))

def validate(val_loader, model, criterion, iter, logger=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target,is_posi) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)[0]
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1,5))

        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            log.l.info(('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5)))

    log.l.info(('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
          .format(top1=top1, top5=top5, loss=losses)))

    return top1.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    filename = '_'.join((args.snapshot_pref, args.modality.lower(), filename))
    torch.save(state, filename)
    if is_best:
        best_name = '_'.join((args.snapshot_pref, args.modality.lower(), 'model_best.pth.tar'))
        shutil.copyfile(filename, best_name)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
    lr = args.lr * decay
    decay = args.weight_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr #* param_group['lr_mult']
        param_group['weight_decay'] = decay #* param_group['decay_mult']


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__=='__main__':
    main()
