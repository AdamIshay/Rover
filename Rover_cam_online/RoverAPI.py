from __future__ import print_function
from rover.Data_cam import *
from overhead_cam1 import *
import os, sys
from rover.Pygame_UI import *
from rover import Rover
import time
import numpy as np
#from scipy.misc import imresize

#should be in Rover_cam os.chdir('/home/mpcr/Adam_python/Rover_cam')

class RoverRun(Rover):
    def __init__(self, fileName, network_name, autonomous, driver, rover, FPS,
                 view, save_data, framework, image_type, normalization,
                 norm_vals, num_out):
        Rover.__init__(self)
        self.FPS = FPS
        self.view = view
        self.speed = 0.5
        self.save_data = save_data
        self.userInterface = Pygame_UI(self.FPS, self.speed)
        self.image = None
        self.quit = False;
        self.angle = 0
        self.autonomous = autonomous
        self.image_type = image_type
        self.im_shp = None
        self.act = self.userInterface.action_dict['q']
        #self.reward_tracker=Reward_Tracker(reward_tokens)
        #self.punish_tracker=Reward_Tracker(punish_tokens)
        self.rover_tracker = rover_tracker(reward_tracker,punish_tracker)


        if self.autonomous is True:
            if self.image_type in ['color', 'Color']:
                self.im_shp = [None, 240, 320, 3]
            elif self.image_type in ['framestack', 'Framestack']:
                self.im_shp = [None, 130, 320, 3]
                self.framestack = np.zeros([1, 130, 320, self.FPS])
                self.stack = [0, 5, 15]
            elif self.image_type in ['grayscale', 'gray', 'Grayscale']:
                self.im_shp = [None, 130, 320, 1]

        self.d = Data(driver, rover, save_data, framework, fileName,
                      network_name, self.im_shp, normalization, norm_vals,
                      num_out, self.image_type)

        if self.autonomous is True:
            self.d.load_network()

        self.run()


    def run(self):
        print('runningggg')
        while type(self.image) == type(None):
            pass
        jj=1
        #cv2.startWindowThread()
        while not self.quit:
            s = self.image
            if self.view is True:
                self.userInterface.feed(s)

       	    key = self.userInterface.getActiveKey()
            if key == 'z':
                self.quit = True
            
            if self.autonomous is not True:
                    if key in ['w', 'a', 's', 'd', 'q', ' ']:
                        self.act = self.userInterface.action_dict[key]
            else:
                s = self.d.normalize(s)
                raw_img=s.copy()
                
                #process image here
                s=self.process_image(s)
                
                self.angle = self.d.predict(s)
                print(self.angle)
                self.act = self.userInterface.action_dict[self.angle]
            print(self.act)
            
            print(jj)
            
            #if 0==0:
            if jj%1==0:
                #to show the video

                img=self.rover_tracker.make_frame() #(480,640,3)
                img = img[:,:,::-1]
                
                larger_rover_img = raw_img.repeat(2, axis=1).repeat(2, axis=0)[:,:,::-1]
                print(type(img),type(larger_rover_img))
                print(img.shape,larger_rover_img.shape)
                
                final_img=np.concatenate((img.transpose((2,0,1)),larger_rover_img.transpose((2,0,1))),axis=2)
                print('final_img shape ',final_img.shape)
                
                vis.image(final_img,win='image')
                
                
                jj=1
            jj+=1
            if self.act[-1] != 9 and self.save_data in ['y', 'Y']:
                
                        self.d.add_data(s[:,:,0], self.angle, self.rover_tracker.net_reward)
                        print('s shape is' + str(s.shape))
            self.set_wheel_treads(self.act[0], self.act[1])
            self.userInterface.manage_UI()

        # cleanup and stop vehicle
        self.set_wheel_treads(0, 0)
        self.userInterface.cleanup()

        # save training data and close
        self.d.save()
        self.close()

    def process_image(self,image):
            out=rgb2gray(image)
            out=resize(out,(75,100,1))
            return out
