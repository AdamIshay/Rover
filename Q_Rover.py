#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 14:50:48 2018

@author: mpcr
"""


import warnings
warnings.filterwarnings('ignore')

import torch
from torch.autograd import Variable
import torchvision.models as models
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
import torch.nn.functional as F
import numpy as np
import os,sys
import h5py
from skimage.transform import resize
import pdb
import time
import copy 
from sklearn import preprocessing
#import matplotlib.pyplot as plt
import pickle



#Classes

#MDP_dataset - for building MDP tuples
#DQN_agent - Deep Q network
#Agent_dataset -  dataset for training DQN

# =============================================================================
# DQN Training process:
#     1. Collect Data from autonomous Q_agent (~50,000 frames) (maybe save resized)
#     2. Run frames through supervised network and store the supervised outputs     
#     3. Use these outputs to reward DQN in training 
#     4. Repeat process with updates Q_agent
# 
# =============================================================================



def progressBar(epoch=0, value=0, endvalue=0, loss=0, sample_output=0, training=True, total_epoch=0, bar_length=20):

        percent = float(value) / endvalue
        arrow = '-' * int(round(percent * bar_length)-1) + '>'
        spaces = ' ' * (bar_length - len(arrow))
        
        sample_output=np.around(np.array(sample_output),2)
        argsort=np.argsort(-sample_output)
        if training == True:
            sys.stdout.write("\r Epoch: {0} / {1} Percent: [{2}] {3}%   loss = {4}               sample_output= {5}    argsort = {6}         ".format(epoch, total_epoch, arrow + spaces, int(round(percent * 100)),round(loss,3),sample_output,argsort))
        else:
            sys.stdout.write("\r Epoch: {0} Percent: [{1}] {2}%".format(epoch, arrow + spaces, int(round(percent * 100))))
        
        sys.stdout.flush()

def preprocess_data_files(path,new_path):
    '''Takes data from Q_agent, changes to grayscale, puts new file in 'path' '''

    Q_data_files=os.listdir(path)
    
    for file in Q_data_files:
        loaded_h5=h5py.File(path+'/'+file,'r+')

        temp_data_x=loaded_h5['X']
        temp_data_x=np.array(temp_data_x)
        temp_data_x=temp_data_x[...,::-1]
        temp_data_x=rgb2gray(temp_data_x)
        
        temp_data_y_new=loaded_h5['Y']
        

        temp_data_x_new=np.zeros((len(temp_data_x),75,100))
        
        
        for frame in range(len(temp_data_x)):
            
            temp_data_x_new[frame]=resize(temp_data_x[frame],(75,100))
            
        os.chdir(path+'_resized')
        with h5py.File('h'+file, 'w') as hf:
            hf.create_dataset("X",  data=temp_data_x_new)
            hf.create_dataset("Y",  data=temp_data_y_new)
        
        loaded_h5.close()
        
        
    os.chdir('/home/mpcr/Adam_python/Rover')
    
    return None

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def get_supervised_rewards(s_net,path):
    
    binarizer=preprocessing.LabelBinarizer()
    binarizer.fit([0.0,1.0,2.0,3.0])
    s_net=s_net
    path=path
    filenames=os.listdir(path);filenames=np.array(filenames)
    frame_accum=[0]
    frame_accum_w_end=[0]
    
    for file in filenames:
        f=h5py.File(path + file, 'r')
        frame_accum.append(f['X'].shape[0]+frame_accum[-1]-1)
        f.close()
    
    for file in filenames:
        f=h5py.File(path + file, 'r')
        frame_accum_w_end.append(f['X'].shape[0]+frame_accum_w_end[-1])
        f.close()

        frame_accum=np.array(frame_accum)
        frame_accum_w_end=np.array(frame_accum_w_end)
        
        total_frames=frame_accum[-1]-len(filenames)



class MDP_dataset_new(Dataset):
    'MDP dataset which takes in filenames '
    def __init__(self,agent,s_net,path,replay_size):
        
        self.agent=agent
        self.replay_size=replay_size
        self.binarizer=preprocessing.LabelBinarizer()
        self.binarizer.fit([-1.0,0.0,1.0,2.0])
        self.s_net=s_net
        self.path=path
        self.filenames=os.listdir(self.path);self.filenames=np.array(self.filenames)
        frame_accum=[0]
        frame_accum_w_end=[0]
        self.index_list=[]
        for file in self.filenames:
            f=h5py.File(path + file, 'r')
            frame_accum.append(f['X'].shape[0]+frame_accum[-1]-1)
            f.close()
        
        for file in self.filenames:
            f=h5py.File(path + file, 'r')
            frame_accum_w_end.append(f['X'].shape[0]+frame_accum_w_end[-1])
            f.close()
        
        self.frame_accum=np.array(frame_accum)
        self.frame_accum_w_end=np.array(frame_accum_w_end)
        
        self.total_frames=self.frame_accum[-1]-len(self.filenames)
        self.progress_counter=0
        
    def get_frame_location(self,index,frame_array):
        
        files_less=1*(index>=frame_array)
        
        file_location=np.max(np.where(files_less==1))

        index_in_file=index-frame_array[file_location]
        
        return file_location,index_in_file

    def get_sample_state(self,index,frame_array):
        
        
        file_location,index_in_file=self.get_frame_location(index=index,frame_array=frame_array)
        f=h5py.File(self.path + self.filenames[file_location], 'r')
 
        X=f['X'][index_in_file]
        label=f['Y'][index_in_file]
        f.close()

        X=X[...,np.newaxis]
        X=np.transpose(X,(2,0,1))
        progressBar(epoch='Loading Replay',value= self.progress_counter//2,endvalue= self.replay_size,training=False)
        self.progress_counter+=1
        
        return X,label
    
    def get_state_next_state(self,index):
        
        #pdb.set_trace()
        state,label=self.get_sample_state(index,self.frame_accum)
        
        next_state,next_label=self.get_sample_state(index+1,self.frame_accum_w_end)
        
        return state,label,next_state
        
    def get_reward(self,action,label):
        
        return reward_matrix[label,action]     
    
    def __len__(self):
        'Denotes the total number of samples'
        return self.total_frames

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        
        state,label,next_state=self.get_state_next_state(index)
        
        s_action_values=self.s_net.forward(torch.from_numpy(state).unsqueeze_(0).type(torch.cuda.FloatTensor))
        s_action_values=torch.tensor(s_action_values.data,requires_grad=False)
        s_action=s_action_values.argmax().data.tolist()
        
        action=int(label)

        reward=self.get_reward(action,s_action)
        
        mdp_list=[state, action, next_state, reward]
        
        self.index_list.append(index)
        
        return mdp_list

class Q_Agent():
    def __init__(self,epsilon=.05,gamma=.9,learning_rate=.001):
        self.DQN=models.resnet34()
        self.DQN.fc=torch.nn.Linear(512,4)     
        self.DQN.conv1=torch.nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.DQN.avgpool=torch.nn.AdaptiveAvgPool2d(1)
        self.DQN.cuda()

        self.epsilon=epsilon
        self.gamma=gamma
        
        self.opt=optim.RMSprop(self.DQN.parameters(),lr=learning_rate)
        
        self.run_type=0
        
    def load_Q_agent(self,filename):
        
        load_path='/home/mpcr/Adam_python/Rover/Q_agents/'+filename
        if os.path.isfile(load_path):
            print("=> loading checkpoint '{}'".format(load_path))
            checkpoint = torch.load(load_path)
            self.DQN.load_state_dict(checkpoint['state_dict'])
            self.opt.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' "
                      .format(load_path))
        else:
            print("=> no checkpoint found at '{}'".format(load_path))
    
        return None

    def save_Q_agent(self,filename):
    
        state = {'state_dict': self.DQN.state_dict(),
                 'optimizer': self.opt.state_dict() }
        torch.save(state, '/home/mpcr/Adam_python/Rover/Q_agents/'+filename)

    def forward(self,x):
        
        if self.run_type=='train':
            self.DQN.train()
        if self.run_type==0:
            sys.exit("ERROR:runtype")
        else:
            self.DQN.eval()
        
        x=self.DQN.forward(x)
        
        return x
    
    def epsilon_greedy(self,x):
        #small number of actions means 1.0 epsilon gives 25% chance to each action
        if np.random.rand()<self.epsilon:
            x=torch.tensor(np.random.rand(4)).view((1,4))
        else:
            
            if self.run_type=='train':
                self.train()
            else:
                self.DQN.eval()
            
            x=self.DQN.forward(x)
        
        return x
        
    def optimize(self,replay_memory):
        pass;
        
class Agent_Dataset(Dataset):
    'Pytorch Dataset for preparing MDP data for training'
    def __init__(self, agent,data_loader,replay_size):
        'Initialization'
        self.agent=agent
        self.agent_target=copy.deepcopy(agent)
        self.agent_target.DQN.cuda(1)
        self.agent_target.DQN.eval()
        self.len=replay_size
        self.data_loader=data_loader
        self.mdp_data=self.create_replay_memory()
        
    def create_replay_memory(self):
        return next(iter(self.data_loader))
    
    def update_target(self):
        self.agent_target=copy.deepcopy(self.agent)
        self.agent_target.DQN.cuda(1)
        return None

    def format_Q_data(self,index):
        
        self.agent.run_type='eval'
        self.agent.DQN.eval()
        self.agent_target.DQN.eval()

        chosen_actions=self.mdp_data[1][index]
        rewards=self.mdp_data[3][index]
        
        states=self.mdp_data[0][index]
        states_next=self.mdp_data[2][index]
        
        total_states=len(states)
        
        Q_values=[]      
        
        states_gpu=states.view(1,1,75,100).float().cuda()
        Q_values=list(self.agent.DQN.forward(states_gpu).detach().cpu())
        
        del states_gpu
        
        Q_values=torch.stack(Q_values)
        target_Q_values=Q_values.clone().cpu()
        
        #set all target q-values the same as predicted, except for the action selected, to only backpropogate for selected action
        
        Q_values_next_max=[]
        
        states_next_gpu=states_next.view(1,1,75,100).float().cuda(1)
        Q_values_next_max=list(self.agent_target.DQN.forward(states_next_gpu).detach().cpu())
        
        Q_values_next_max=Variable(torch.stack(Q_values_next_max).data,requires_grad=False)
        
        Q_value_next_max=Q_values_next_max.max(dim=1)[0]
        
        chosen_actions_np=chosen_actions.numpy()
        
        target_Q_value=rewards.float()+self.agent.gamma*Q_value_next_max.cpu()
        target_Q_values[0,chosen_actions_np.astype(np.int).item()]= target_Q_value
        target_Q_values=Variable(target_Q_values.data,requires_grad=False)
        
        self.X=states
        self.target_Q_values=target_Q_values
        
        return None

    def __len__(self):
        'Denotes the total number of samples'
        return self.len

    def __getitem__(self,index):
        'Gets replay_memory sized batch from the MDP list'
        # Select sample
        self.format_Q_data(index)
        return self.X, self.target_Q_values
        #return self.X[index],self.target_Q_values[index]



def train_DQN(agent,Q_data_loader,agent_data, criterion, epochs,batch_size,opt,target_update):
    
    opt.zero_grad()

    loss_hist=[]
    #mean_hist=[]
    sample_hist=[]

    for epoch in range(epochs):  
        n_of_batches=agent_data.len//batch_size
        agent.run_type='train'
        agent.DQN.train()
        
        print(Agent.opt)
        for i_batch,batch in enumerate(Q_data_loader):
            total_batches=epoch*n_of_batches + i_batch
            #pdb.set_trace()
            #target update
            if total_batches%target_update==0:
                agent_data.update_target()
             
            opt.zero_grad()

            x=batch[0].float().cuda()
            y=batch[1].view(batch_size, 4).float().cuda()
            outputs=agent.DQN.forward(x)
            loss = criterion(outputs,y)
            loss.backward()
            
            loss_hist.append(loss.data.item())
            agent.DQN.eval()
            sample_hist.append(agent.DQN(x[0:1]).data.tolist())
            agent.DQN.train()
            
            progressBar(epoch, i_batch, n_of_batches,loss=loss_hist[-1], sample_output=sample_hist[-1],training=True,total_epoch=epochs)
            opt.step()
            
    return loss_hist,sample_hist

def exp_lr_scheduler(optimizer, epoch, lr_decay=0.1, lr_decay_epoch=7):
    """Decay learning rate by a factor of lr_decay every lr_decay_epoch epochs"""
    epoch+=1;
    if epoch % lr_decay_epoch:
        return optimizer

    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr_decay
    return optimizer

def create_replay_memory(data_loader,size):
    replay_memory=[]
    
    for i in range(size):
        replay_memory.append(next(iter(data_loader)))
    
    return replay_memory

def load_model(file_path):
    #instantiate model
    s_net=models.resnet34()
    s_net.fc=torch.nn.Linear(512, 4)
    s_net.conv1=torch.nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
    s_net.avgpool=torch.nn.AdaptiveAvgPool2d(1)
    #load parameters in 
    s_net.load_state_dict(torch.load(file_path))
    
    s_net.cuda()

    return s_net

def save_object(obj,filename):
    #pickle.dump(obj, filename, pickle.HIGHEST_PROTOCOL)
    os.chdir('/home/mpcr/Adam_python/Rover/Q_agents')
        
    with open(filename, "wb") as f:
        pickle.dump(obj,f)

    os.chdir('/home/mpcr/Adam_python/Rover')



if __name__ == "__main__":
    
    #reward matrix: if i is s_net output argmax and j was q_net output argmax 
    #is reward_matrix(i,j)
    
    reward_matrix=np.array([[13,-3.5,-.5,-5.5],[0,2.8,0,-5.0],[-.5,-3.5,1.7,-5.5],[-.5,-.25,-.5,.75]])
    
    #loading the saved s_net

    s_net=load_model(file_path='/home/mpcr/Adam_python/Rover/gray19_noweights_3epochs')
    s_net.cuda()
    s_net.eval()
    
    #location of Q_agent data
    
    Q_data_path='/home/mpcr/Adam_python/Rover/Q_data_gen1_resized/'
    all_filenames=os.listdir(Q_data_path)

    #either load previous Q_agent or create one
    
    Agent=Q_Agent(epsilon=.9,gamma=.8,learning_rate=.01)
    Agent.run_type='eval'
    
    Agent.load_Q_agent('gen_1_20_working')
    
    
    #path='/home/mpcr/Adam_python/Rover/Q_agent_data_resized/'
    
    replay_size=30000
    
    #batch_size of data_loader should be ~32, the size of this will limit how frequent update_target will take effect
    batch_size=300
    
    training_batch_size=300 #batch_size should be a factor of replay_size
    
    #replay memory only stored in mdp_dataset_new
    
    all_data=MDP_dataset_new(agent=Agent,s_net=s_net,path=Q_data_path,replay_size=replay_size)
    
    mdp_data_loader=DataLoader(dataset=all_data,batch_size=replay_size,shuffle=True) 
    
    agent_data=Agent_Dataset(agent=Agent,data_loader=mdp_data_loader,replay_size=replay_size) #
    
    Q_data_loader=DataLoader(dataset=agent_data,batch_size=batch_size,shuffle=True)
    
    
    criterion=torch.nn.SmoothL1Loss()
    
    loss_hist,sample_hist=train_DQN(agent=Agent,Q_data_loader=Q_data_loader, agent_data=agent_data,criterion=criterion,epochs=25,batch_size=training_batch_size,opt=Agent.opt,target_update=20)
    
    Agent.save_Q_agent('gen_2_3')
    
    


