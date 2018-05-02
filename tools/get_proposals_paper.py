import os,sys
import time,argparse
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import numpy as np
from watershed import Watershed
sys.path.append('ops/')
from utils import *


LEN_VIDEO_CUBE=16
#CLIPS=[0.0005]#,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.09,0.07,0.05,0.03,0.01]
alter_vec=[4,10,20,30,40,50,60,70,80,90,100]
thre=[0.4,0.3,0.2,0.1,0.05,0.01,0.005,0.001]
NUM_CLASSES=21

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

def smooth(scores_list,len_linear):
    new_scores=np.zeros_like(scores_list)

    smooth_vec=np.array([1]*len_linear);len_smooth=len_linear;sum_smooth=sum(smooth_vec)
    old_score=np.concatenate([scores_list[0].reshape(1,-1).repeat(len_smooth/2,0),\
                              scores_list,\
                              scores_list[-1].reshape(1,-1).repeat((len_smooth/2)-1,0)])
    for i in range(len(new_scores)):
        new_scores[i]=np.dot(old_score[i:i+len_smooth].T,smooth_vec)/sum_smooth
    return new_scores

def Post(scores,post_process,len_linear):
    scores=np.array(scores)

    if post_process=='softmax':
        new_scores=softmax(scores)
    if post_process=='sigmoid':
        new_scores=sigmoid(scores)
    if not len_linear==0:
        new_scores=smooth(new_scores,len_linear)
    return np.array(new_scores)

def vis(scores_video,vide0o):
    scores_show=np.zeros([len(scores_video),2])
    scores_show[:,0]=scores_video[:,0]
    scores_show[:,1]=scores_show[:,1]*5+scores_video[:,2]*5+scores_video[:,3]
    plt.plot(scores_show)
    plt.title(video)
    plt.show()

def add_scores(proposals,scores_all_videos_class_dic):

    new_proposals=list()
    for proposal in proposals:
        video,s,e=proposal[0],proposal[1],proposal[2]
        score_tmp=np.sum(scores_all_videos_class_dic[video][s:e,:],0)/float(e-s)
        score_inds=np.argmax(score_tmp);score=score_tmp[score_inds]
        if score_inds==0:continue
        new_proposals.append([video,s,e,score_inds,score])
    return new_proposals

def clip_proposals(proposals,prior_infos):
    new_proposals=list()
    for proposal in proposals:
        id=proposal[3]
        gap=proposal[2]-proposal[1]
        if gap in range(prior_infos['min_l'][id-1],prior_infos['max_l'][id-1]):
            new_proposals.append(proposal)
    return new_proposals

def write_result(proposals,fps,path):
    out_put=list()
    for proposal in proposals:
        video,s,e,i,c=proposal[0],proposal[1],proposal[2],proposal[3],proposal[4]
        out_='{} {} {} {} {}'.format(video,s/float(fps[video]),e/float(fps[video]),i,c)
        out_put.append(out_)
    with open('tmp_results/{}'.format(path),'w') as fw:fw.write('\n'.join(out_put))

def get_args():
    parser=argparse.ArgumentParser(description='watershed algorithm')
    parser.add_argument('score_file',type=str,
                        default='tmp_results/P3D_rgb_16frames.pkl')
    parser.add_argument('--file_list',default='../metadata/final_test_list_perframe.txt')
    parser.add_argument('-f',type=str,
                        default='../metadata/final_test_list.txt')
    parser.add_argument('-v',action='store_true')
    parser.add_argument('-pp',type=str,default='softmax')
    parser.add_argument('--len_linear',default=36,type=int)
    parser.add_argument('-c','-candidate',type=str,default='watershed')
    args=parser.parse_args()
    
    return args

if __name__=='__main__':
    global args
    args=get_args()



    ins_names,video_labels,n_frames,video_names,ids=get_infos(args.file_list)
    video_names=get_video_names(args.file_list)
    info_dic=joblib.load('../metadata/all_infos_test.pkl')
    video_infos={'videos':info_dic['videos'],
             'video_fps_dic':info_dic['video_fps_dic'],
             'actions':info_dic['actions'],
             'video_l_dic':info_dic['video_l_dic']}

    score_files=joblib.load(args.score_file)
    print('getting the result from given path')


    #scores_class=dict()
    scores_class={video_name:score_raw for (video_name,score_raw) in zip(video_names,score_files['score'])}
    print('read input data and build dicts done!')
    # if args.p=='together':
    #     scores_proposal=scores_class
    # else:
    #     scores_proposal=scores_raw[0]

    '''
    video_names : ['video_test_0000131_121_137', '...', '...'] # type is : video_test_id_start_end
    '''
    info_dic=joblib.load('../metadata/all_infos_test.pkl')
    video_infos={'videos':info_dic['videos'],
                 'video_fps_dic':info_dic['video_fps_dic'],
                 'actions':info_dic['actions'],
                 'video_l_dic':info_dic['video_l_dic']}
    prior=joblib.load('../test-code/action_infos_dic_val.pkl')
    prior_infos={'max_l':prior['max_l'],
                 'min_l':prior['min_l'],
                 'mean_l':prior['mean_l']}
    print('read infos done!')

    scores_all_videos_class=list()
    scores_all_videos_class_dic=dict()

    for i,video in enumerate(video_infos['videos']):
        print('====>It\'s the video {} '.format(video))
        #video_cubes=[_ for _ in video_names if video in video_names]
        #video_inds=range(1,video_infos['video_l_dic'][video]-15,8)
        #video_cubes=['{}_{}_{}'.format(video,video_inds[i],video_inds[i]+16) for i in range(len(video_inds))]
        video_inds=['{}_{}'.format(_,_+1) for _ in range(1,video_infos['video_l_dic'][video])]
        video_cubes=['{}_{}'.format(video,video_inds[i]) for i in range(len(video_inds))]

        scores_video_class=Post([scores_class[_] for _ in video_cubes],args.pp,args.len_linear)

        # scores_proposal_video=Post([scores_proposal[_] for _ in video_cubes],args.pp)
        if args.v:vis(scores_video_class,video)
        
        scores_all_videos_class.append([scores_video_class,video])
        scores_all_videos_class_dic[video]=scores_video_class

    print('scores process finished, now computing candidates...')

    num0=0
    for i1 in alter_vec:
        for i2 in thre:

            if args.c=='watershed':
                Watershed_Ins=Watershed('smooth',NUM_CLASSES)
                proposals=Watershed_Ins.get_proposals(scores_all_videos_class_dic,alter=(i1,i2)) # get proposals to generate candidates.



            print('adding scores...')
            proposals=add_scores(proposals,scores_all_videos_class_dic)
            print('clipping proposals from priors...')
            proposals=clip_proposals(proposals,prior_infos)
            joblib.dump(proposals,'proposals.pkl',protocol=2)


            write_result(proposals,video_infos['video_fps_dic'],path='final_result_{}'.format(num0))
            num0+=1
            print('i1={},i2={},num0={}'.format(i1,i2,num0))