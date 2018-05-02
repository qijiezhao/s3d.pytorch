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
        for i,line in enumerate(lines):
            list_line=line.strip().split(' ')
            name='{}_{}_{}'.format(list_line[0].split('/')[-1],list_line[2],list_line[3])
            posi=1 if not list_line[5]=='0' else 0
            label=np.array([int(_) for _ in list_line[4].split('+')])*posi
            video_labels[i,label]=1#label
            ins_names.append(name)
            n_frames.append(int(list_line[1]))
            ids.append(list_line[2])
            if not list_line[0].split('/')[-1] in video_names:
                video_names.append(list_line[0].split('/')[-1])
    return ins_names,video_labels,n_frames,video_names,ids

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

def Post(scores_list):
    scores_list=np.array(scores_list)
    new_scores=np.zeros_like(scores_list)

    smooth_vec=np.array([1.5,2,2,2.5,2.5,3,3,4,4,3,3,2.5,2.5,2,2,1.5]);len_smooth=16;sum_smooth=sum(smooth_vec)
    old_score=np.concatenate([scores_list[0].reshape(1,-1).repeat(len_smooth/2,0),\
                              scores_list,\
                              scores_list[-1].reshape(1,-1).repeat((len_smooth/2)-1,0)])
    for i in range(len(new_scores)):
        new_scores[i]=np.dot(old_score[i:i+len_smooth].T,smooth_vec)/sum_smooth
    return new_scores

parser = argparse.ArgumentParser(description='This script is to compute mAP for per frame score')
parser.add_argument('score_files',type=str,default=None)
parser.add_argument('--score_weights',nargs='+',type=float,default=None)
parser.add_argument('--file_list',default='../metadata/final_test_list_perframe.txt')
parser.add_argument('--post',default='softmax',type=str)
parser.add_argument('--inter',default=True,type=bool)
args=parser.parse_args()

ins_names,video_labels,n_frames,video_names,ids=get_infos(args.file_list)
info_dic=joblib.load('../metadata/all_infos_test.pkl')
video_infos={'videos':info_dic['videos'],
             'video_fps_dic':info_dic['video_fps_dic'],
             'actions':info_dic['actions'],
             'video_l_dic':info_dic['video_l_dic']}

score_files=joblib.load(args.score_files)

print('getting the result from given path')
weights=args.score_weights
score_list=list()

score_list=np.array(score_files['score'])
score_dic={ins_name:score for ins_name,score in zip(ins_names,score_list)}

new_scores=[]
for video in video_names:
    n_frames=video_infos['video_l_dic'][video]
    video_tmp_names=['{}_{}_{}'.format(video,_,_+1) for _ in range(1,n_frames)]
    video_score=Post([score_dic[video_tmp_name] for video_tmp_name in video_tmp_names])
    new_scores.extend(video_score)
    print('=====> video {} done'.format(video))

score_list=np.array(new_scores)
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

# lb.fit(range(n_class))
# label_list=lb.transform(label_list)

APs=[0]

print('BG: {}'.format(average_precision_score(label_list[:,0],score_list[:,0])))

for i in range(1,n_class,1):
    APs.append(average_precision_score(label_list[:,i],score_list[:,i]))


print('\n'.join(['Action {} gets AP: {}'.format(Actions[i],APs[i]) for i in range(1,n_class,1)]))
print('\n =====> mAP is :\n\t\t {}'.format(np.mean(APs[1:])))
