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


# use only a subset of the data at a time

    # make MDPs
    # states, next states = frames
    #action = -1,0,1,2
    #rewards = 1 if the DQN chooses the right action, 0 otherwise


# =============================================================================
# 9/5 - Idea: let the q-agent run until it hits the wall, then terminate manually.
# The agent's potential to learn is a function of how well the supervised network 
# gives it feedback. If the agent hits wall in testing phase, it will likely suck.
# it might be a good idea to retrain supervised net to perform better, and add
# additional training data of the correct behavior of the rover when it hits the
# wall.
# 
#     - Also, retrain on images which are black and white and resized to ~(75,100), this
# will allow us to fit a large amount (>50k) images into the replay buffer. Also,
# use two gpus, one for the Q-network and one for the target network. 
# =============================================================================


    
#Q_agent controls rover and movements are recorded

# =============================================================================
# DQN Training process:
#     1. Collect Data from autonomous Q_agent (~50,000 frames) (maybe save resized)
#     2. Run frames through supervised network and store the supervised outputs     
#     3. Use these outputs to reward DQN in training 
#     4. Repeat process with updates Q_agent
# 
# =============================================================================


def preprocess_data_files():
    '''Takes data from Q_agent, changes to grayscale, puts new file in 'path' '''
    
    path='/home/mpcr/Adam_python/Rover/Q_agent_data_resized'
    
    Q_data_files=os.listdir(path)
    
    for file in Q_data_files:
        loaded_h5=h5py.File(path+'/'+file,'r+')

        #pdb.set_trace()
        temp_data_x=loaded_h5['X']
        temp_data_x=np.array(temp_data_x)
        temp_data_x=temp_data_x[...,::-1]
        temp_data_x=rgb2gray(temp_data_x)
        
        temp_data_y_new=loaded_h5['Y']
        
        #pdb.set_trace()
        
        temp_data_x_new=np.zeros((len(temp_data_x),75,100))
        
        
        for frame in range(len(temp_data_x)):
            
            temp_data_x_new[frame]=resize(temp_data_x[frame],(75,100))
            
        os.chdir(path+'_resized')
        with h5py.File('h'+file, 'w') as hf:
            hf.create_dataset("X",  data=temp_data_x_new)
            hf.create_dataset("Y",  data=temp_data_y_new)
        
        
        
        
        
    
        
        #loaded_h5.create_dataset('X',data=temp_data_x_new)

        
        loaded_h5.close()
        
        
    os.chdir('/home/mpcr/Adam_python/Rover')
    
    return None

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def get_supervised_rewards(s_net,path):
    
    
    binarizer=preprocessing.LabelBinarizer()
    binarizer.fit([-1.0,0.0,1.0,2.0])
    s_net=s_net
    path=path
    filenames=os.listdir(path);filenames=np.array(filenames)
    frame_accum=[0]
    frame_accum_w_end=[0]
    
    for file in filenames:
        f=h5py.File(path + file, 'r')
        frame_accum.append(f['X'].shape[0]+frame_accum[-1]-1)
        f.close()
    #frame_accum= np.delete(frame_accum,0)
    
    for file in filenames:
        f=h5py.File(path + file, 'r')
        frame_accum_w_end.append(f['X'].shape[0]+frame_accum_w_end[-1])
        f.close()
        
        
        
        
        frame_accum=np.array(frame_accum)
        frame_accum_w_end=np.array(frame_accum_w_end)
        
        total_frames=frame_accum[-1]-len(filenames)

    pdb.set_trace()







class MDP_dataset_new(Dataset):
    'MDP dataset which takes in filenames '
    def __init__(self,agent,s_net,path,replay_size):
        #pdb.set_trace()
        #self.filenames=np.array(os.listdir('/home/mpcr/Right2'))
        
        
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
        #frame_accum= np.delete(frame_accum,0)
        
        for file in self.filenames:
            f=h5py.File(path + file, 'r')
            frame_accum_w_end.append(f['X'].shape[0]+frame_accum_w_end[-1])
            f.close()
        
        
        
        
        self.frame_accum=np.array(frame_accum)
        self.frame_accum_w_end=np.array(frame_accum_w_end)
        
        self.total_frames=self.frame_accum[-1]-len(self.filenames)

# =============================================================================
#     def get_rewards(self):
#         Q_data_path='/home/mpcr/Adam_python/Rover/Q_agent_data_resized/'
#         
#         
#         Q_filenames=os.listdir(Q_data_path)
#         
# =============================================================================
        
        
    def get_frame_location(self,index,frame_array):
        
        files_less=1*(index>=frame_array)
        
        file_location=np.max(np.where(files_less==1))

        index_in_file=index-frame_array[file_location]
        
        
        
        return file_location,index_in_file

    def get_sample_state(self,index,frame_array):
        
        #pdb.set_trace()
        
        file_location,index_in_file=self.get_frame_location(index=index,frame_array=frame_array)

        f=h5py.File('/home/mpcr/Adam_python/Rover/Q_agent_data_resized/' + self.filenames[file_location], 'r')
        #a_group_key = list(f.keys())[0]
        
        X=f['X'][index_in_file]
        label=f['Y'][index_in_file]
        f.close()
        #pdb.set_trace()
        #X=resize(X,(224,224,3))
        X=X[...,np.newaxis]
        X=np.transpose(X,(2,0,1))

        
        return X,label

        
    
    def get_state_next_state(self,index):
        
        #pdb.set_trace()
        state,label=self.get_sample_state(index,self.frame_accum)
        
        next_state,next_label=self.get_sample_state(index+1,self.frame_accum_w_end)
        
        return state,label,next_state
        
    def get_reward(self,action,label):
        
        if action==label:
            return 1
        else:
            return 0
        
    
    def __len__(self):
        'Denotes the total number of samples'
        return self.total_frames

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        #pdb.set_trace()
        
        state,label,next_state=self.get_state_next_state(index)
        
        s_action_values=self.s_net.forward(torch.from_numpy(state).unsqueeze_(0).type(torch.cuda.FloatTensor))
        s_action_values=torch.tensor(s_action_values.data,requires_grad=False)
        s_action=s_action_values.argmax().data.tolist()
        
        #pdb.set_trace()
        
        action_values=self.agent.forward(torch.from_numpy(state).unsqueeze_(0).type(torch.cuda.FloatTensor))
        action_values=torch.tensor(action_values.data,requires_grad=False)
        action=action_values.argmax().data.tolist()
        
        reward=self.get_reward(action,s_action)
        
        mdp_list=[state, action, next_state, reward]#,label, s_action_values,action_values]
        
        self.index_list.append(index)
        
        return mdp_list







class MDP_dataset(Dataset):
    'MDP dataset which takes in filenames '
    def __init__(self,filenames,agent,s_net):
        #pdb.set_trace()
        #self.filenames=np.array(os.listdir('/home/mpcr/Right2'))
        self.filenames=np.array(filenames)
        self.agent=agent
        self.binarizer=preprocessing.LabelBinarizer()
        self.binarizer.fit([-1.0,0.0,1.0,2.0])
        self.s_net=s_net
        frame_accum=[0]
        frame_accum_w_end=[0]
        
        for file in self.filenames:
            f=h5py.File('/home/mpcr/Right2/' + file, 'r')
            frame_accum.append(f['X'].shape[0]+frame_accum[-1]-1)
            f.close()
        #frame_accum= np.delete(frame_accum,0)
        
        for file in self.filenames:
            f=h5py.File('/home/mpcr/Right2/' + file, 'r')
            frame_accum_w_end.append(f['X'].shape[0]+frame_accum_w_end[-1])
            f.close()
        
        
        
        
        self.frame_accum=np.array(frame_accum)
        self.frame_accum_w_end=np.array(frame_accum_w_end)
        
        self.total_frames=self.frame_accum[-1]-len(self.filenames)





    def get_sample_state(self,index,frame_array):
        
        #pdb.set_trace()
        
        files_less=1*(index>=frame_array)
        
        file_location=np.max(np.where(files_less==1))

        index_in_file=index-frame_array[file_location]

        f=h5py.File('/home/mpcr/Right2/' + self.filenames[file_location], 'r')
        #a_group_key = list(f.keys())[0]
        
        X=f['X'][index_in_file]
        label=f['Y'][index_in_file]
        f.close()
        X=resize(X,(224,224,3))
        X=np.transpose(X,(2,0,1))

        
        return X,label

        
    
    def get_state_next_state(self,index):
        
        #pdb.set_trace()
        state,label=self.get_sample_state(index,self.frame_accum)
        
        next_state,next_label=self.get_sample_state(index+1,self.frame_accum_w_end)
        
        return state,label,next_state
        
    def get_reward(self,action,label):
        
        if action==label:
            return 1
        else:
            return 0
        
    
    def __len__(self):
        'Denotes the total number of samples'
        return self.total_frames

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        #pdb.set_trace()
        
        state,label,next_state=self.get_state_next_state(index)
        
        s_action_values=self.s_net.forward(torch.from_numpy(state).unsqueeze_(0).type(torch.cuda.FloatTensor))
        s_action_values=torch.tensor(s_action_values.data,requires_grad=False)
        s_action=s_action_values.argmax().data.tolist()
        
        action_values=self.agent.forward(torch.from_numpy(state).unsqueeze_(0).type(torch.cuda.FloatTensor))
        action_values=torch.tensor(action_values.data,requires_grad=False)
        action=action_values.argmax().data.tolist()
        
        reward=self.get_reward(action,s_action)
        
        mdp_list=[state, action, next_state, reward]#,label, s_action_values,action_values]
        
        
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
    
        return None #model, optimizer
        




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
            x=torch.tensor(np.random.rand(4))
        else:
            
            if self.run_type=='train':
                self.train()
            else:
                self.eval()
            

            x=self.forward(x)
        
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
        
    def create_replay_memory(self):
# =============================================================================
#         replay_memory=[]
#         
#         for i in range(self.len):
#             replay_memory.append(next(iter(self.data_loader)))
#         
#         return replay_memory
# =============================================================================
        return next(iter(self.data_loader))
    
    def update_target(self):
        self.agent_target=copy.deepcopy(agent)
        self.agent_target.cuda(1)
        return None
    
    
    def format_Q_data(self):
        #pdb.set_trace()
        self.agent.run_type='eval'
        self.agent.DQN.eval()
        self.agent_target.DQN.eval()
        #self.agent_target.agent.eval()
        
        
        #get replay memory ~50k MDP lists
        mdp_data=self.create_replay_memory()
        
        
        #mdp_data=create_replay_memory(data_loader=self.data_loader, size=self.len)
        
        chosen_actions=mdp_data[1]
        rewards=mdp_data[3]
        #rewards=np.array(self.agent.memory[:,3],dtype=np.float32)
        
        states=mdp_data[0]
        states_next=mdp_data[2]
        
        total_states=len(states)
        
        batch_size_Q=40
        
        total_iter=total_states//batch_size_Q
        leftover=total_states%batch_size_Q
        
        Q_values=[]
        #pdb.set_trace()
        for i in range(total_iter):
            states_gpu=states[i*batch_size_Q:(i+1)*batch_size_Q].float().cuda()
            Q_values=Q_values+list(self.agent.DQN.forward(states_gpu))
        
# =============================================================================
#         states_gpu=states[(i+1)*batch_size_Q:].float().cuda()
#         Q_values=Q_values+list(self.agent.DQN.forward(states_gpu))
# =============================================================================
        
        del states_gpu
        
        
        #boards_unrolled_next=np.hstack(self.agent.memory[:,2]).T
        #boards_unrolled_next=torch.tensor(boards_unrolled_next,dtype=torch.float32)
        
        #Q_values=self.agent.agent.forward(boards_unrolled) # batch_size x 7
        
        
        Q_values=torch.stack(Q_values)
        target_Q_values=Q_values.clone().cpu()
        
        #set all target q-values the same as predicted, except for the action selected, to only backpropogate for selected action
        
# =============================================================================
#         below needs to be changed to run through target network 
# =============================================================================
        Q_values_next_max=[]
        
        for i in range(total_iter):
            states_next_gpu=states_next[i*batch_size_Q:(i+1)*batch_size_Q].float().cuda(1)
            Q_values_next_max=Q_values_next_max+list(self.agent_target.DQN.forward(states_next_gpu))
        
        #pdb.set_trace()
# =============================================================================
#         states_next_gpu=states_next[(i+1)*batch_size_Q:].float().cuda()
#         Q_values_next_max=Q_values_next_max+list(self.agent.DQN.forward(states_next_gpu))
# =============================================================================
        
        
        #Q_value_next_max=self.agent.agent.forward(boards_unrolled_next).max(dim=1)[0]
        #Q_value_next_max=self.agent_target.forward(boards_unrolled_next).max(dim=1)[0]
        
        Q_values_next_max=Variable(torch.stack(Q_values_next_max).data,requires_grad=False)
        
        Q_value_next_max=Q_values_next_max.max(dim=1)[0]
        
        chosen_actions_np=chosen_actions.numpy()
        

        target_Q_value=rewards.float()+self.agent.gamma*Q_value_next_max.cpu()
        target_Q_values[np.arange(len(chosen_actions_np)),chosen_actions_np.astype(np.int)]= target_Q_value
        target_Q_values=Variable(target_Q_values.data,requires_grad=False)
        
        self.X=states
        self.target_Q_values=target_Q_values
        
        return None

    def __len__(self):
        'Denotes the total number of samples'
        return self.len

    def __getitem__(self):
        'Gets replay_memory sized batch from the MDP list'
        # Select sample
        self.format_Q_data()
        return self.X, self.target_Q_values
        #return self.X[index],self.target_Q_values[index]



def train_DQN(agent,agent_data, criterion, total_replays,batch_size,opt):
    
    
    opt.zero_grad()
    

    val_hist=[]
    
    
    #pdb.set_trace()
    for replay in range(total_replays):
        X,Y=agent_data.__getitem__() # index value is not used 
        pdb.set_trace()
        n_of_batches=len(X)//batch_size
        #print(f'replay: {replay}')
        agent.run_type='train'
        agent.DQN.train()
        
        for b in range(n_of_batches):
            
            opt.zero_grad()
            
            x=X[b*batch_size:(b+1)*batch_size].float().cuda()
            y=Y[b*batch_size:(b+1)*batch_size].float().cuda()
            outputs=agent.DQN.forward(x)
            
            loss = criterion(outputs,y)
            loss.backward()
            #pdb.set_trace()
            #print(f'        sample: {outputs[0]}    loss: {loss.detach().cpu().numpy().round(2)}')
            
            
            opt.step()
            
# =============================================================================
#         opt.zero_grad()
#         
#         x=X[(b+1)*batch_size:].float().cuda()
#         y=Y[(b+1)*batch_size:].float().cuda()
#         
#         outputs=agent.DQN.forward(x)
#         
#         loss = criterion(outputs,y)
#         loss.backward()
#         opt.step()
# =============================================================================
        
        










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



    






if __name__ == "j":

    #loading the saved s_net
    
    s_net=load_model(file_path='/home/mpcr/Adam_python/Rover/gray19_noweights')
    s_net.cuda()
    s_net.eval()
    
    #location of Q_agent data
    all_filenames=os.listdir('/home/mpcr/Adam_python/Rover/Q_agent_data_resized')

    #either load previous Q_agent or create one
    
    Agent=Q_Agent(epsilon=.95,gamma=.9,learning_rate=.0001)
    Agent.run_type='eval'
    
    #Agent.load_Q_agent('filename')
    
    
    path='/home/mpcr/Adam_python/Rover/Q_agent_data_resized/'

    all_data=MDP_dataset_new(agent=Agent,s_net=s_net,path=path,replay_size='sfasj')

    
    replay_size=400
    training_batch_size=40 #batch_size should be a factor of replay_size
    data_loader=DataLoader(dataset=all_data,batch_size=replay_size,shuffle=True)
    
    
    agent_data=Agent_Dataset(agent=Agent,data_loader=data_loader,replay_size=replay_size) #
    

    
    #criterion=torch.nn.CrossEntropyLoss()  
    criterion=torch.nn.MSELoss()  
    
    train_DQN(agent=Agent,agent_data=agent_data,criterion=criterion,total_replays=5,batch_size=training_batch_size,opt=sample_agent.opt)
    

    
    





#cut first 5 images and labels off to get rid of black frames

# =============================================================================
#     
# for file in all_filenames:
#     
#     f=h5py.File('/home/mpcr/Right2/' + file, 'r+')
#     
#     temp_data_x=f['X']
#     temp_data_x=temp_data_x[5:]
#     
#     temp_data_y=f['Y']
#     temp_data_y=temp_data_y[5:]
#     
#     
#     
#     del f['X']
#     del f['Y']
#     f.create_dataset('X',data=temp_data_x)
#     f.create_dataset('Y',data=temp_data_y)
#     
#     f.close()
#     
# =============================================================================




#epsilon_greedy not being used? 
    
#gray and then resize is faster 
    
# =============================================================================
# import time
# def test_time(iterations):
#     time_hist=[]
#     for i in s:
#         
#         tic=time.time()
#         something=resize(i,(75,100))
#         something=rgb2gray(something)
#         
#         
#         toc=time.time()
#         time_hist.append(toc-tic)
#     return time_hist
# =============================================================================




