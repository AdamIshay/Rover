
import os
import warnings
warnings.filterwarnings('ignore')


import time
import numpy as np
import h5py
import progressbar
import datetime
import tflearn
from tflearn.layers.core import input_data
import torchvision.models as models
from NetworkSwitch import *
import torch
import torch.nn as nn
#from scipy.misc import imresize
from skimage.transform import resize
import pdb

sys.path.insert(0,'/home/mpcr/Adam_python/Rover_cam')

from Q_Rover_cam import Q_Agent

#sys.path.insert(0,'/home/mpcr/Adam_python/Rover')



class Data():
    def __init__(self, driver_name, rover_name, save_data, framework,
                 filename, network_name, input_shape, normalization, norm_vals,
                 num_out, image_type):
        
        
        self.mem_size=300 #self.mem_size must be a multiple of self.mem_update_size
        self.new_chunk_index=0
        self.mem_update_size=30 
        self.mem_full=False
        
        
        self.angles = np.zeros((self.mem_size))
        self.images = np.zeros((self.mem_size,75,100))
        self.rewards=np.zeros((self.mem_size))
        self.temp_angles=[]
        self.temp_images=[]
        self.temp_rewards=[]


        
        self.start = time.time()
        self.names = driver_name + '_' + rover_name
        self.save_data = save_data
        self.framework = framework
        self.filename = filename
        self.network_name =network_name
        self.input_shape = input_shape
        self.normalization = normalization
        self.norm_vals = norm_vals
        self.num_out = num_out
        self.image_type = image_type


    def load_network(self):
        if self.framework in ['tf', 'TF']:
            if self.network_name in ['ResNet34',
                                     'ResNet26',
                                     'ResNeXt34',
                                     'ResNeXt26']:
                tflearn.config.init_training_mode()
            self.network_name = modelswitch[self.network_name]
            self.network = input_data(shape=self.input_shape)
            self.network = self.network_name(self.network,
                                             self.num_out,
                                             drop_prob=1.0)
            self.model = tflearn.DNN(self.network)
            self.model.load(self.filename)

        elif self.framework in ['PT', 'pt' ,  'ptgray']:
            self.network_name = models.__dict__[self.network_name]
            self.model=self.network_name()
            

                
            self.model.conv1=torch.nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1,
                           bias=False)
            self.model.avgpool=torch.nn.AdaptiveAvgPool2d(1)
            
            self.model.fc = nn.Linear(512, self.num_out)
            self.model.cuda()
            self.model.load_state_dict(torch.load(self.filename))
            self.model.eval()
            
        elif self.framework in ['ptQ']:
            'load in the Q_network class so epsilongreedy can be used'
            #pdb.set_trace() # epsilon .99 gen0
            self.Agent=Q_Agent(epsilon=.99,gamma=.9,learning_rate=.0001)
           # self.Agent.load_Q_agent(self.network_name)
            #self.Agent.DQN.eval()
            self.Agent.load_Q_agent(self.filename)
            self.Agent.DQN.eval()
            self.Agent.run_type='eval'
            
            
        return


    def predict(self, s):
        if self.framework in ['tf', 'TF']:
            #s = s[None, 110:, ...]
            s = s[None,...]
            if self.image_type in ['grayscale', 'framestack']:
                s = np.mean(s, 3, keepdims=True)
                if self.image_type in ['framestack']:
                    current = s
                    self.framestack = np.concatenate((current,
                                               self.framestack[:, :, :, 1:]), 3)
                    s = self.framestack[:, :, :, self.stack]
            out = self.model.predict(s)
        
        
        elif self.framework in ['pt', 'PT']:
            out = resize(s, (224, 224)).transpose((2, 0, 1))[None,...]
            out = torch.from_numpy(out).float().cuda()
            out = self.model(out).detach().cpu().numpy()[0, :]
        
        elif self.framework in ['ptgray','ptQ']:
            
            
            
            #take process out
            
# =============================================================================
#             out=rgb2gray(s)
#             out=resize(out,(75,100,1))
#             out=np.transpose(out,(2,0,1))[None,...]
#             out = torch.from_numpy(out).float().cuda()
# =============================================================================
            
            
            
            out=np.transpose(s,(2,0,1))[None,...]
            out = torch.from_numpy(out).float().cuda()
            

            if self.framework == 'ptQ':
                out=self.Agent.epsilon_greedy(out).detach().cpu().numpy()[0, :]
            else:
                out = self.model(out).detach().cpu().numpy()[0, :]
        


        print(out)

        return np.argmax(out)

    def normalize(self, x):
        if self.normalization is not None:
            if self.normalization == 'instance_norm':
                x = (x - np.mean(x)) / (np.std(x) + 1e-6)
            elif self.normalization == 'channel_norm':
                for j in range(x.shape[-1]):
                    x[..., j] -= self.norm_vals[j]
        return x


    def add_data(self, image, action, reward):
        self.temp_angles.append(action)
        self.temp_images.append(image)
        self.temp_rewards.append(reward)
        
        print('length of self.temp_angles is ' + str(len(self.temp_angles)))

        
        print('mem_full = ' + str(self.mem_full))
        print('new_chunk_index = ' + str(self.new_chunk_index))
        
        print('Collecting Data')
        
        print(self.angles[:150])
        
        if len(self.temp_angles)==self.mem_update_size:
            #pdb.set_trace()
            
            temp_angles_np=np.array(self.temp_angles)
            temp_images_np=np.array(self.temp_images)
            temp_rewards_np=np.array(self.temp_rewards)[:,0]
            
            print(self.new_chunk_index)
            print(self.new_chunk_index+self.mem_update_size)
            
            print(self.angles[self.new_chunk_index:self.new_chunk_index+self.mem_update_size].shape)
            print(temp_angles_np.shape)
            print(temp_images_np.shape)
            print(temp_rewards_np.shape)
            
            self.angles[self.new_chunk_index:self.new_chunk_index+self.mem_update_size]=temp_angles_np
            self.images[self.new_chunk_index:self.new_chunk_index+self.mem_update_size]=temp_images_np
            self.rewards[self.new_chunk_index:self.new_chunk_index+self.mem_update_size]=temp_rewards_np
            
            self.temp_angles=[]
            self.temp_images=[]
            self.temp_rewards=[]
        
        
            if self.new_chunk_index !=self.mem_size-self.mem_update_size:
                self.new_chunk_index+=self.mem_update_size
            else:
                self.mem_full=True
                self.new_chunk_index=0
        
        
        
        return
    
    
    
    def load_replay_memory(self):
        pass;

    def save(self):
        if self.save_data in ['y', 'Y', 'yes', 'Yes']:
            print('Saving the Training Data you collected.')
            self.images = np.array(self.images, dtype='uint8')
            self.angles = np.array(self.angles, dtype='float16')
            self.rewards=np.array(self.rewards, dtype = 'float32')
            elapsedTime = int(time.time() - self.start)
            dset_name = str(elapsedTime) + "seconds_" + self.names + ".h5"
            os.chdir('replay_memory')
            h5f = h5py.File(dset_name, 'w')
            h5f.create_dataset('X', data=self.images)
            h5f.create_dataset('Y', data=self.angles)
            h5f.create_dataset('R', data=self.rewards)
            h5f.close()
            os.chdir('..')
        return

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])