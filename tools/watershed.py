import os,sys
import time
import numpy as np
from IPython import embed
from operator import itemgetter


class Watershed(object):
    def __init__(self,type_='clips',num_classes=2):
        self.type=type_
        self.num_classes=num_classes

    def get_proposals(self,video_score_dic,alter=None):
        if alter==None:return 0
        if self.type=='clips':
            return self.get_proposals_clips(video_score_dic,alter)
        elif self.type=='smooth':
            return self.get_proposals_smooth(video_score_dic,alter)

    def get_proposals_smooth(self,video_score_dic,alter):
        proposals=list()
        smooth_term,thre=alter
        for smooth_term in [smooth_term]:
            '''
            clips=[0.95,0.5,0.25,0.15,0.05,0.01]
            '''
            for i,(video,score) in enumerate(video_score_dic.items()):
                tmp_score=score.copy()
                tmp_score=self.smooth(tmp_score,smooth_term)
                tmp_score[tmp_score<thre]=0
                tmp_score[tmp_score>thre]=1
                start_end_list=self.get_s_e(tmp_score,video)
                proposals.extend(start_end_list)

            print('smooth term {} has done!'.format(smooth_term))
        return proposals
    def get_proposals_clips(self,video_score_dic,alter):
        proposals=list()
        for clip in alter:
            '''
            clips=[0.95,0.5,0.25,0.15,0.05,0.01]
            '''
            for i,video,score in enumerate(video_score_dic.items()):
                tmp_score=score.copy()
                tmp_score[tmp_score<clip]=0
                tmp_score[tmp_score>clip]=1
                start_end_list=self.get_s_e(tmp_score,video)
                proposals.extend(start_end_list)

            print('clip {} has done!'.format(clip))
        return proposals

    def smooth(self,old_score,terms):
        smoothing_vec=np.ones(terms)
        sum_smooth_vec=terms
        new_scores=np.zeros_like(old_score)
        old_score=np.concatenate([old_score[0].reshape(1,-1).repeat(len(smoothing_vec)/2,0),\
                                 old_score,\
                                 old_score[-1].reshape(1,-1).repeat(len(smoothing_vec)/2-1,0)])  # padding with repeat
        for i in range(len(new_scores)):
            new_scores[i]=np.dot(old_score[i:i+len(smoothing_vec)].T,smoothing_vec)/sum_smooth_vec
        return new_scores

    def get_s_e(self,score_ins,video):
        s_e_list=list()
        for i in range(1,self.num_classes):
            s,e=0,0;lock=0
            score_item=score_ins[:,i] # each class
            score_item=np.array([0]+list(score_item)+[0])
            for j in range(len(score_item)):
                if lock==0 and score_item[j]!=0:
                    s=j
                    lock=1
                if lock==1 and score_item[j]==0:
                    e=j
                    s_e_list.append([video,s,e,i])
                    lock=0
        return s_e_list
        #return self.post(s_e_list,score_ins,video) # to ensemble by curves


    def post(self,s_e_list,score_ins,video):
        posted_s_e_list=s_e_list
        for ii in range(1,self.num_classes):
            tmp_s_e_lists=[_ for _ in s_e_list if _[3]==ii]
            s_s=[_[1] for _ in tmp_s_e_lists]
            e_s=[_[2] for _ in tmp_s_e_lists]

            for i,s_ in enumerate(s_s):
                for j,e_ in enumerate(e_s):
                    if i<j and s_<e_:
                        if sum(score_ins[s_:e_,ii])/float((e_-s_))>0.9:
                            posted_s_e_list.append([video,s_,e_,ii])
        return posted_s_e_list

