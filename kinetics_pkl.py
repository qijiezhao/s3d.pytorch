import torch.utils.data as data

from PIL import Image
import os
import sys
import os.path
import random
import numpy as np
import torch
import cPickle
from cStringIO import StringIO
import collections
import base64
import random

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def pil_loader(buf,is_gray):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    if isinstance(buf, str):
        tempbuff = StringIO()
        tempbuff.write(buf)
        tempbuff.seek(0)
        img = Image.open(tempbuff)
    elif isinstance(buf,collections.Sequence):
        img = Image.open(StringIO(buf[-1]))
    else:
        img = Image.open(StringIO(buf))
    return img.convert('L') if is_gray else img.convert('RGB')


def video_loader(is_train, frames, nsegment, seglen, imageloader,is_gray):
    offset = 0
    nsample= nsegment
    videolen=len(frames)
    average_dur = videolen / nsegment

    ret = []
    if videolen <= 0:
        return ret

    for i in range(nsample):
        idx = 0
        if is_train:
            if average_dur >= seglen:
                idx = random.randint(0, average_dur-seglen)
                idx += i * average_dur
            elif average_dur >= 1:
                idx += i*average_dur
            else:
                idx = 0
        else:
            if average_dur >= seglen:
                idx = (average_dur-seglen) / 2
                idx += i * average_dur
            elif average_dur >= 1:
                idx += i*average_dur
            else:
                idx = 0

        for j in range(idx,idx+seglen):
            imgbuf=frames[j % videolen]
            img = imageloader(imgbuf,is_gray)
            ret.append(img)

    return ret


def default_loader(path,is_gray):
    return pil_loader(path,is_gray)

def load_list(infile):
    ret=[]
    with open(infile) as f:
        for line in f:
            ret.append(line.strip())
    random.shuffle(ret)
    return ret


class KineticsPkl(data.Dataset):

    def __init__(self, filelist, nseg, seglen, is_train, cropsize, transform=None, target_transform=None,
                 loader=default_loader, use_gray=False):
        self.seglen = seglen
        self.nseg = nseg
        self.videos = load_list(filelist)
        self.transform = transform
        self.cropsize = cropsize
        self.target_transform = target_transform
        self.loader = loader
        self.is_train = is_train
        self.use_gray = use_gray

    def __getitem__(self, index):
        channel_num = 1 if self.use_gray else 3
        try:
            videop = self.videos[index]

            dataloaded = cPickle.load(open(videop))
            vid = dataloaded[0]
            label = dataloaded[1]
            frames = dataloaded[2]

            ret = torch.FloatTensor(self.nseg,self.seglen*channel_num,self.cropsize,self.cropsize).zero_()
            target = -1

            imgs = video_loader(self.is_train, frames, self.nseg, self.seglen, self.loader, self.use_gray)

            if len(imgs) == 0:
                return ret, target, vid

            ret = self.transform(imgs)
            target = int(label)

            if self.target_transform is not None:
                target_batch = self.target_transform(target_batch)
    
            return ret, target,vid

        except:
            #print "exception, return zeros and ignored target indices"
            ret = torch.FloatTensor(self.nseg,self.seglen*channel_num,self.cropsize,self.cropsize).zero_()
            return ret,-1,''

    def __len__(self):
        return len(self.videos)
