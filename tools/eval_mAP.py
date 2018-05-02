from __future__ import print_function
import argparse
import sys,os
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn import preprocessing
from sklearn.externals import joblib
from IPython import embed
lb=preprocessing.LabelBinarizer()

Actions=['BG','BaseballPitch','BasketballDunk','Billiards','CleanAndJerk','CliffDiving',\
         'CricketBowling','CricketShot','Diving','FrisbeeCatch','GolfSwing','HammerThrow','HighJump',\
         'JavelinThrow','LongJump','PoleVault','Shotput','SoccerPenalty','TennisSwing',
         'ThrowDiscus','VolleyballSpiking']

def get_infos(path):
    ins_names=list()
    video_names=list()
    n_frames=list()
    ids=list()
    with open(path,'r')as fp:
        lines=fp.readlines()
        video_labels=np.zeros([len(lines),21],dtype=int)
        posis=np.zeros([len(lines),4],dtype=int)
        for i,line in enumerate(lines):
            list_line=line.strip().split(' ')
            name='{}_{}_{}'.format(list_line[0].split('/')[-1],list_line[2],list_line[3])
            posi=1 if not list_line[5]=='0' else 0
            label=np.array([int(_) for _ in list_line[4].split('+')])*posi
            video_labels[i,label]=1#label
            ins_names.append(name)
            n_frames.append(int(list_line[1]))
            ids.append(list_line[2])
            video_names.append(list_line[0].split('/')[-1])

            if '+' in list_line[5]:
                posi=np.array([int(_) for _ in list_line[5].split('+')])
            else:
                posi=np.array(int(list_line[5]))
            posis[i,posi]=1
    return ins_names,video_labels,n_frames,video_names,ids,posis

def softmax(z):
    assert len(z.shape) == 2, len(z.shape)
    s = np.max(z, axis=1)
    s = s[:, np.newaxis] # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis] # dito
    return e_x / div

def sigmoid(z):
    assert len(z.shape) == 2, len(z.shape)
    return 1/(np.exp(-z)+1)


def ensemble(inds,names,weights,scores):
    name=names[7]
    id_names=names==name
    if id_names[0]==False:
        id_min=list(id_names).index(True)
        inds[0:id_min]=inds[id_min]
    elif id_names[15]==False:
        id_min=list(id_names).index(False)
        inds[id_min:]=inds[id_min-1]
    else:
        pass
    return np.dot(weights,scores[inds])

def interpolation(scores,weights,infos):
    new_scores=list()
    len_scores=len(scores)

    for i in range(len_scores):
        inds=np.array(range(i-7,i+9,1))
        inds[inds<0]=0
        inds[inds>=len_scores]=len_scores-1
        new_scores.append(ensemble(inds,np.array(infos['video_names'])[inds],weights,scores))
        if i%100==0:print(i)
    return new_scores

parser = argparse.ArgumentParser(description='This script is to compute mAP for per frame score')
parser.add_argument('score_files',nargs='+',type=str,default=None)
parser.add_argument('--score_weights',nargs='+',type=float,default=None)
parser.add_argument('--file_list',default='../metadata/final_test_list.txt')
parser.add_argument('--post',default='softmax',type=str)
parser.add_argument('--inter',default=True,type=bool)
args=parser.parse_args()

ins_names,video_labels,n_frames,video_names,ids,posis=get_infos(args.file_list)
score_files=[joblib.load(_) for _ in args.score_files]

print('getting the result from given path')
weights=args.score_weights
score_list=list()
for ins_name in ins_names:
    if ins_name in score_files[0]['class'].keys():
        tmps=None
        for i in range(len(score_files)):
            if i==0:
                tmps=weights[0]*score_files[i]['class'][ins_name]
                continue
            tmps=weights[i]*score_files[i]['class'][ins_name]+tmps
        score_list.extend([tmps.tolist()])
embed()
score_list=np.array(score_list)
if args.post=='softmax':
    score_list=softmax(score_list)
elif args.post=='sigmoid':
    score_list=sigmoid(score_list)
label_list=video_labels[:len(score_list)]

n_class=21
# recs=[]
# for i in range(0,n_class,1):
#     inds=label_list==i
#     ss=score_list[inds]
#     rec=sum(np.argmax(ss,1)==i)/float(len(ss))
#     print(Actions[i],rec,len(ss),np.sum([np.argmax(score_list,1)==i]))
#     recs.append(rec)
# print(np.mean(rec))
# embed()
if False:#True:
    length=4
    print('computing interpolation results')
    #weights=[1]*length
    weights=[3,5,5,3]
    # score_list=interpolation(score_list,weights,
    #                          {'video_names':video_names,'video_labels':label_list,'n_frames':n_frames,'ids':ids})
    new_scores=list()
    len_scores=len(score_list)

    for i in range(len_scores):
        inds=np.array(range(int(i-(length/2)+1),int(i+(length/2)+1),1))
        inds[inds<0]=0
        inds[inds>=len_scores]=len_scores-1
        names=np.array(video_names)[inds]
        name=names[int(length/2)]
        id_names=names==name
        if id_names[0]==False:
            id_min=list(id_names).index(True)
            inds[0:id_min]=inds[id_min]
        elif id_names[length-1]==False:
            id_min=list(id_names).index(False)
            inds[id_min:]=inds[id_min-1]
        rst=np.dot(weights,score_list[inds])

        new_scores.append(rst)
        if i%100==0:print(i)
    score_list=np.array(new_scores)
    np.save('ensembled_data.npy',score_list)

# lb.fit(range(n_class))
# label_list=lb.transform(label_list)

APs=[0]

print('BG: {}'.format(average_precision_score(label_list[:,0],score_list[:,0])))

for i in range(1,n_class,1):
    APs.append(average_precision_score(label_list[:,i],score_list[:,i]))

print('\n'.join(['Action {} gets AP: {ap:.03f}'.format(Actions[i],ap=APs[i]) for i in range(1,n_class,1)]))
print('\n =====> mAP is :\n\t\t {map:.4f}'.format(map=np.mean(APs[1:])))


for i in range(1,4,1):
    APs=[0]

    print('computing the {} sensitive area'.format(i))
    inds=posis[:,i]==1
    label_list_tmp=label_list[inds]
    score_list_tmp=score_list[inds]
    for j in range(1,n_class):
        APs.append(average_precision_score(label_list_tmp[:,j],score_list_tmp[:,j]))

    print('\n'.join(['Action {} gets AP: {ap:.03f}'.format(Actions[j],ap=APs[j]) for j in range(1,n_class,1)]))
    print('\n =====> mAP is :\n\t\t {map:.4f}'.format(map=np.mean(APs[1:])))