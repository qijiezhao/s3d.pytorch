import os,sys
import numpy as np,matplotlib.pyplot as plt
import argparse
import time


def get_video_result(list_file,id):
    num0=0
    s=0;e=0
    s_lock=0;e_lock=0
    scores_out=[]
    lines=open(list_file,'rb').readlines()
    file_pre=lines[0].split('/')[4]
    for num,line in enumerate(lines[1:]):
        file_=line.split('/')[4]
        if not file_==file_pre:
            num0+=1
            file_pre=file_
        if num0==id and s_lock==0:
            s=num
            s_lock=1
        if num0==id+1 and e_lock==0:
            e=num
            e_lock=1
        if num0==id:
            score=1 if int(line.strip().split(' ')[-1])==args.action else 0
            scores_out.append(score)
    return s,e,scores_out

def draw_lines(gt,scores_chosen):
    fig=plt.figure(figsize=(30,6))
    colors=['red','blue']
    len_data=len(gt)
    plt.plot(gt,c=colors[0],label='gt',)
    plt.plot(scores_chosen,c=colors[1],label='scores')
    plt.legend(loc='best')
    plt.ylim(ymax=2)
    plt.show()

def softmax(z):
    assert len(z.shape) == 2, len(z.shape)
    s = np.max(z, axis=1)
    s = s[:, np.newaxis] # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis] # dito
    return e_x / div

if __name__=='__main__':
    parser=argparse.ArgumentParser(description='this script helps drawing a actionness map for testing score')
    parser.add_argument('--score_file',default='tmp/thumos14_resnet152_rgb.npz',type=str)
    parser.add_argument('--test_list',default='../metadata/test_list_frames.txt',type=str)
    parser.add_argument('--id',default=1,type=int)
    parser.add_argument('--action',default=1,type=int)
    global args
    args=parser.parse_args()

    s_,e_,gt = get_video_result(args.test_list,args.id)

    scores=np.load(args.score_file)['scores']
    scores_chosen=softmax(scores[s_:e_])[:,args.action]

    if len(scores_chosen)>gt:
        scores_chosen=scores_chosen[:len(gt)]
    else:
        gt=gt[:len(scores_chosen)]


    draw_lines(gt,scores_chosen)


