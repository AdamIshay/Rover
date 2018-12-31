# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import warnings
warnings.filterwarnings('ignore')


import torch
import torchvision.models as models
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
import torch.nn.functional as F
import numpy as np
import os
import h5py
from skimage.transform import resize
import pdb
import time
from sklearn import preprocessing
from sklearn.utils.class_weight import compute_class_weight
import sys

import warnings
warnings.filterwarnings('ignore')


torch.cuda.is_available()

#  0  : straight
# -1 : left
#  1 : right
#  2 : backwards

#load data

# =============================================================================
# f=h5py.File('/home/mpcr/Right2/Run_148seconds_Chris_Sheri.h5', 'r')
# keys = list(f.keys())
# data=f[keys[0]]
# 
# =============================================================================

def Train_supervised(model,dataset,data_loader,data_loader_val,criterion,learning_rate,optimizer,num_epochs): 
    
    opt=optimizer
    opt.zero_grad()
    val_hist=[]
    toc=0
    for epoch in range(num_epochs):
        model.train()
        for i_batch,batch in enumerate(data_loader):
            
            opt.zero_grad()
            
            outputs=model.forward(batch[0].type(dtype))
            targets=batch[1].type(torch.cuda.LongTensor)
            
            loss = criterion(outputs,targets)
            accuracy=get_accuracy(outputs,targets)
            loss.backward()
            opt.step()
            tic=time.time()
            progressBar(epoch=epoch,value=i_batch,endvalue=dataset.frame_accum[-1]//batch_size, val_loss=float(loss),val_acc=accuracy,total_epoch=num_epochs)
            toc=time.time()
        model.eval()
        
        val_acc_list=[]
        for i_batch_val,batch_val in enumerate(data_loader_val):
         
            val_outputs=model.forward(batch_val[0].type(dtype))
            val_targets=batch_val[1].type(torch.cuda.LongTensor)
            val_acc=get_accuracy(val_outputs,val_targets)
            val_acc_list.append(val_acc)
            val_loss=criterion(val_outputs,val_targets)
            
            avg_val_acc=np.array(val_acc_list).mean()
            
            val_hist.append(avg_val_acc)
            
        print(f'\n {i_batch} / {dataset.frame_accum[-1]//batch_size} time/batch:{round((tic-toc),2)} val_loss: {val_loss}  val_acc {round(avg_val_acc,2)} \n')
    
    return val_hist

def progressBar(epoch=0, value=0, endvalue=0, val_loss=0, val_acc=0, total_epoch=0, bar_length=20):

        percent = float(value) / endvalue
        arrow = '-' * int(round(percent * bar_length)-1) + '>'
        spaces = ' ' * (bar_length - len(arrow))

        
        sys.stdout.write("\r Epoch: {0} / {1} Percent: [{2}] {3}%   loss = {4}      acc={5}".format(epoch, total_epoch, arrow + spaces, int(round(percent * 100)),round(val_loss,3),round(val_acc,3)))
       
        sys.stdout.flush()

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


def get_accuracy(outputs,targets):
    #pdb.set_trace()
    correct_bool=outputs.max(dim=1)[1]==targets
    accuracy=correct_bool.sum().double()/len(correct_bool)
    return accuracy.data.tolist()

class Rover_Dataset(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self,filenames):

        self.filenames=np.array(filenames)
        self.binarizer=preprocessing.LabelBinarizer()
        self.binarizer.fit([0.0,1.0,2.0,3.0])
        frame_accum=[0]
        for file in self.filenames:
            f=h5py.File('/home/mpcr/Right2/' + file, 'r')
            frame_accum.append(f['X'].shape[0]+frame_accum[-1])
        self.frame_accum=np.array(frame_accum)
        
        self.total_frames=self.frame_accum[-1]

    def get_sample(self,index):
        
        files_less=1*(index>=self.frame_accum)
        
        file_location=np.max(np.where(files_less==1))

        index_in_file=index-self.frame_accum[file_location]

        f=h5py.File('/home/mpcr/Right2/' + self.filenames[file_location], 'r')
        
        X=f['X'][index_in_file]
        Y=f['Y'][index_in_file]
        X=X[...,::-1]
        
        X=rgb2gray(X)
        X=resize(X,(75,100,1))
        X=np.transpose(X,(2,0,1))
        
        self.X=X
        self.label=self.binarizer.transform([Y])
        self.label=self.label.argmax()
    def __len__(self):
        'Denotes the total number of samples'
        return self.total_frames

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        self.get_sample(index)
        
        return self.X,self.label


if __name__ == "__main__":
    
    torch.cuda.is_available()
    dtype = torch.cuda.FloatTensor

    all_filenames=os.listdir('/home/mpcr/Right2')
    
    train_data_filenames=all_filenames[:-1]    
    val_data_filenames=all_filenames[-1:]
    
    # compute model weights
    
    f=h5py.File('/home/mpcr/Right2/' + all_filenames[1], 'r')
    
    class_weights=compute_class_weight(class_weight='balanced',classes=np.unique(f['Y']),y=f['Y'])
    
    f.close()
    
    s_net=models.resnet34()
    s_net.fc=torch.nn.Linear(512, 4)
    s_net.conv1=torch.nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
    s_net.avgpool=torch.nn.AdaptiveAvgPool2d(1)

    s_net.cuda()
    
    # create Dataset object

    train_data=Rover_Dataset(train_data_filenames)
    val_data=Rover_Dataset(val_data_filenames)
    
    batch_size=1000
    lr=.00001
    n_of_epochs=3
    opt = optim.Adam(s_net.parameters(), lr)
     
    criterion=torch.nn.CrossEntropyLoss()   

    data_loader=DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True,num_workers=5)
    data_loader_val=DataLoader(dataset=val_data,batch_size=50,shuffle=True,num_workers=5)
    
    val_hist=Train_supervised(model=s_net,dataset=train_data,data_loader=data_loader,data_loader_val=data_loader_val,criterion=criterion,learning_rate=lr,optimizer=opt,num_epochs=n_of_epochs)
    
    torch.save(s_net.state_dict(), '/home/mpcr/Adam_python/Rover/gray19_noweights_3epochs')
