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
    

    # set to train mode
    
    val_hist=[]
    toc=0
    for epoch in range(num_epochs):
        model.train()
        #for i_batch,(batch,batch_val) in enumerate(zip(data_loader,data_loader_val)):
        for i_batch,batch in enumerate(data_loader):
            
            
            #pdb.set_trace()
            opt.zero_grad()
            
            outputs=model.forward(batch[0].type(dtype))
            targets=batch[1].type(torch.cuda.LongTensor)
            #targets=batch[1].view(batch[1].shape[0],4)
            
            loss = criterion(outputs,targets)
            
            accuracy=get_accuracy(outputs,targets)
            loss.backward()
            opt.step()
            tic=time.time()
            print(f'{i_batch} / {dataset.frame_accum[-1]//batch_size} time/batch:{round((tic-toc),2)} loss: {loss}   train_acc: {round(accuracy,2)}')
            toc=time.time()
        model.eval()
        
        val_acc_list=[]
        #pdb.set_trace()
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
# =============================================================================
#             
#             if i_batch % 5==0:
#                 
#                 val_outputs=model.forward(batch_val[0].type(dtype))
#                 val_targets=batch_val[1].type(torch.cuda.LongTensor)
#                 val_acc=get_accuracy(val_outputs,val_targets)
#                 
#                 
#                 tic=time.time()
#                 print(f' {i_batch} / {dataset.frame_accum[-1]//batch_size} time/batch:{round((tic-toc),2)} loss: {loss}   train_acc: {round(accuracy,2)}  val_acc {round(val_acc,2)}')
#                 toc=time.time()
#             else:
#                 tic=time.time()
#                 print(f' {i_batch} / {dataset.frame_accum[-1]//batch_size} time/batch:{round((tic-toc),2)} loss: {loss}   train_acc: {round(accuracy,2)}')
#                 toc=time.time()
# =============================================================================
            #print(f' {i_batch} / {dataset.frame_accum[-1]//batch_size}  loss: {loss}   train_acc: {round(accuracy,2)}')
            #print(f'epoch {1+i}/{epochs}')
        #print(f'loss: {float(loss)}')
    
    
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
        #pdb.set_trace()
        #self.filenames=np.array(os.listdir('/home/mpcr/Right2'))
        self.filenames=np.array(filenames)
        
        self.binarizer=preprocessing.LabelBinarizer()
        self.binarizer.fit([-1.0,0.0,1.0,2.0])
        
        frame_accum=[0]
        
        
        for file in self.filenames:
            f=h5py.File('/home/mpcr/Right2/' + file, 'r')
            frame_accum.append(f['X'].shape[0]+frame_accum[-1])
        #frame_accum= np.delete(frame_accum,0)
        self.frame_accum=np.array(frame_accum)
        
        self.total_frames=self.frame_accum[-1]

    def get_sample(self,index):

        #pdb.set_trace()
        
        files_less=1*(index>=self.frame_accum)
        
        file_location=np.max(np.where(files_less==1))

        index_in_file=index-self.frame_accum[file_location]

        f=h5py.File('/home/mpcr/Right2/' + self.filenames[file_location], 'r')
        #a_group_key = list(f.keys())[0]
        
        X=f['X'][index_in_file]
        Y=f['Y'][index_in_file]
        X=X[...,::-1]
        
        #X=resize(X,(224,224,3))
        
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
    #train_data_filenames=all_filenames[:1]
    
    val_data_filenames=all_filenames[-1:]
    
    
    
    
    # compute model weights
    
    
    f=h5py.File('/home/mpcr/Right2/' + all_filenames[1], 'r')
    
    
    class_weights=compute_class_weight(class_weight='balanced',classes=np.unique(f['Y']),y=f['Y'])
    
    f.close()
    
    
    
    
    
    
    
    
    
    
    
    
# =============================================================================
#     removed = list(res_net.children())[:-1]
#     s_net= torch.nn.Sequential(*self.removed)
#     
#     
#     s_net = torch.nn.Sequential(s_net, torch.nn.Linear(2048,4))
# =============================================================================
    
    
    
    #change number of output features, rgb-gray (filter 3-->1), and use
    #adaptive avg pooling to allow for different sized image inputs
    
    s_net=models.resnet34()
    s_net.fc=torch.nn.Linear(512, 4)
    s_net.conv1=torch.nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
    s_net.avgpool=torch.nn.AdaptiveAvgPool2d(1) # test 
    
    
    
    s_net.cuda()
    
    #pdb.set_trace()
    # create Dataset object
    

    
    train_data=Rover_Dataset(train_data_filenames)
    val_data=Rover_Dataset(val_data_filenames)
    
    
    batch_size=1000
    lr=.00003
    n_of_epochs=19
    opt = optim.Adam(s_net.parameters(), lr)
    
    #criterion=torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights,dtype=torch.float,device='cuda'))   
    criterion=torch.nn.CrossEntropyLoss()   
    
    
    data_loader=DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True,num_workers=5)
    data_loader_val=DataLoader(dataset=val_data,batch_size=50,shuffle=True,num_workers=5)
    
    
    val_hist=Train_supervised(model=s_net,dataset=train_data,data_loader=data_loader,data_loader_val=data_loader_val,criterion=criterion,learning_rate=lr,optimizer=opt,num_epochs=n_of_epochs)
    
    torch.save(s_net.state_dict(), '/home/mpcr/Adam_python/Rover/gray19_noweights')
    
# =============================================================================
#     rgb_model=models.resnet34()
#     rgb_model.fc=torch.nn.Linear(512,4)
#     #rgb_model.load_state_dict(torch.load('RGB_model1.pt'))
#     rgb_model.cuda()
# =============================================================================
    
# =============================================================================
#     
#     
#     #validation 
#     
#     val_output_list=[];val_target_list=[];val_acc_list=[];val_hist=[]
#     #pdb.set_trace()
#     rgb_model.zero_grad()
#     for i_batch_val,batch_val in enumerate(data_loader_val):
# 
# 
#             #pdb.set_trace()
#             
#             val_outputs=rgb_model.forward(batch_val[0].type(dtype))
#             val_output_list.append(val_outputs.detach());
#             val_targets=batch_val[1].type(torch.cuda.LongTensor)
#             val_target_list.append(val_targets)
#             val_acc=get_accuracy(val_outputs,val_targets)
#             val_acc_list.append(val_acc)
#             val_loss=criterion(val_outputs,val_targets)
# 
#             avg_val_acc=np.array(val_acc_list).mean()
# 
#             val_hist.append(avg_val_acc)
#     
#     
#     
#     
#     aaa=np.array([[ 238,   42,    0,    1],
#        [ 111, 4179,  411,    1],
#        [   2,  107, 1143,    0],
#        [   0,    0,    0,    0]])
# 
# 
# =============================================================================

# =============================================================================
# 
# 
# 
# #loading the saved s_net
#     
# #instantiate model class 
#     model=models.resnet34()
#     
#   
#     s_net.fc=torch.nn.Linear(512, 4)
#     
#     s_net.cuda()
#     
# #load parameters in 
#     model.load_state_dict(torch.load('/home/mpcr/Adam_python/Rover/saved_model'))
# 
# 
# 
# 
# =============================================================================






    
# =============================================================================
#   
#   # ... after training, save your model accuracy
#   model.save_state_dict('mytraining.pt')
#   torch.save(model.state_dict(),'model_name.pt')
#   # .. to load your previously training model:
#   model.load_state_dict(torch.load('mytraining.pt'))
#
#   to save model use torch.save('path/filename')
#
# =============================================================================

    '''
    
    
    
    for i in range(replay_mem):
        #print(agent1.gamma)
        g.play_game(n_of_turns=2000)
# =============================================================================
#         agent1_data=Agent_Dataset(agent=agent1,agent_target=agent1_target)
#         agent2_data=Agent_Dataset(agent=agent2,agent_target=agent2_target)
#         agent1_data.format_Q_data()
#         agent2_data.format_Q_data()
# =============================================================================
        
        if replay_mem % 1==0:
            agent1.optimize(2,200,agent1_target)
        else:
            agent2.optimize(2,200,agent2_target)
        
        if replay_mem % targ_update==0:
            agent1_target=copy.deepcopy(agent1);agent1_target.run_type='eval'
            agent2_target=copy.deepcopy(agent2);agent2_target.run_type='eval'
# =============================================================================
#         print('agent1 lr:' + str(agent1.opt.param_groups[0]['lr']))
#         print('agent2 lr:' + str(agent2.opt.param_groups[0]['lr']))
#         
#         print('agent1 epsilon:' + str(agent1.epsilon))
#         print('agent2 epsilon:' + str(agent2.epsilon))
# =============================================================================
        
        print(f'replay_mem {1+i}/{replay_mem}............epsilon: {agent1.epsilon}......learning_rate: {agent1.opt.param_groups[0]["lr"]}....gamma:{agent1.gamma}')
        
        agent1.opt.param_groups[0]['lr']+=learning_rate_step
        agent2.opt.param_groups[0]['lr']+=learning_rate_step
        
        agent1.epsilon+=epsilon_step
        agent2.epsilon+=epsilon_step
        
    '''