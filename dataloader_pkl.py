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

IMG_EXTENSIONS=[
     '.jpg', '.JPG', '.jpeg', '.JPEG',
     '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def pil_loader(buf,is_gray):
    if isinstance(buf,str):
        tempbuff=StringIO()
        tempbuff.write(buf)
        tempbuff.seek(0)
        img=Image.open(tempbuff)
    elif isinstance(buf,collections.Sequence):
        img=Image.open(StringIO(buf[-1]))
    else:
        img=Image.open(StringIO(buf))
    return img.convert('L') if is_gray else img.convert('RGB')

def video_loader(is_train,frames,seglen,imageloader,is_gray):
    offset=0
    nsample=seglen
    videolen=len(frames)
    ret=[]

    if videolen<=0:
        return ret

    istart=random.randint(0,videolen-seglen)
    for i in range(istart,istart+seglen):
        imgbuf=frames[i]
        img=imageloader(imgbuf,is_gray)
        ret.append(img)
    return ret

def default_loader(path,is_gray):
    return pil_loader(path,is_gray)

def load_list(infile):
    ret=[]
    with open(infile)as f:
        for line in f:
            ret.append(line.strip())
    random.shuffle(ret)
    return ret


class KineticsPKL(data.Dataset):
    def __init__(self,filelist,seglen,is_train,cropsize,transform=None,target_transform=None,loader=default_loader,use_gray=False):
        self.seglen=seglen
        self.videos=load_list(filelist)
        self.transform=transform
        self.cropsize=cropsize
        self.target_transform=target_transform
        self.loader=loader
        self.is_train=is_train
        self.use_gray=use_gray

    def __getitem__(self,index):
        channel_num=1 if self.use_gray else 3
        try:
            videop=self.videos[index] 
            dataloaded=cPickle.load(open(videop))
            vid=dataloaded[0]
            label=dataloaded[1]
            frames=dataloaded[2]

            ret=torch.FloatTensor(self.seglen,channel_num,self.cropsize,self.cropsize).zero_()
            target=-1

            imgs=video_loader(self.is_train,frames,self.seglen,self.loader,self.use_gray)
            if len(imgs)==0:
                return ret,target,vid
            ret=self.transform(imgs)
            target=int(label)

            if self.target_transform is not None:
                target_batch=self.target_transform(target_batch)

            return ret, target, vid

        except:
            ret=torch.FloatTensor(channel_num,self.seglen,self.cropsize,self.cropsize).zero_()
            return ret,-1,''


    def __len__(self):
        return len(self.videos)
