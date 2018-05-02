import torch
import numpy as np
from sklearn.metrics import confusion_matrix,average_precision_score

def get_video_names(path):
    names=list()
    with open(path,'r')as fp:
        lines=fp.readlines()
        for line in lines:
            line_list=line.strip().split(' ')
            names.append('{}_{}_{}'.format(line_list[0].split('/')[-1],line_list[2],line_list[3]))
    return names
    
def get_grad_hook(name):
    def hook(m, grad_in, grad_out):
        print((name, grad_out[0].data.abs().mean(), grad_in[0].data.abs().mean()))
        print((grad_out[0].size()))
        print((grad_in[0].size()))

        print((grad_out[0]))
        print((grad_in[0]))

    return hook


def softmax(scores):
    es = np.exp(scores - scores.max(axis=-1)[..., None])
    return es / es.sum(axis=-1)[..., None]


def log_add(log_a, log_b):
    return log_a + np.log(1 + np.exp(log_b - log_a))


def class_accuracy(prediction, label):
    cf = confusion_matrix(prediction, label)
    cls_cnt = cf.sum(axis=1)
    cls_hit = np.diag(cf)

    cls_acc = cls_hit / cls_cnt.astype(float)

    mean_cls_acc = cls_acc.mean()

    return cls_acc, mean_cls_acc

def get_AP_video(rst,label,gt_labels):
    gt_labels_to1=np.array([1 if (_==label and label!=0) else 0 for _ in gt_labels])
    rst_to1=np.array(rst[_][label] for _ in range(rst))

    AP=average_precision_score(gt_labels_to1,rst_to1)
    return AP

def get_PSP(source_segment,target_segment):
    max_min=max(source_segment[0],target_segment[0])
    min_max=min(source_segment[1],target_segment[1])
    if min_max<=max_min:
        return 0
    else:
        return(min_max-max_min)/float(source_segment[1]-source_segment[0])
    

def PSP_oneVSall(source_segment,target_segments):
    '''
    source_segment is one segment
    target_segments is multi-segments
    return the max PSP

    '''
    max_PSP=0
    for posi_segment in target_segments:
        PSP=get_PSP(source_segment,posi_segment)
        if PSP>max_PSP:
            max_PSP=PSP
    return max_PSP