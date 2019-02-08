import cv2                            # importing Python OpenCV
import imutils
import numpy as np
import pdb
import matplotlib.pyplot as plt 
import visdom
import pickle
from rover_track_rewards import *
import time 
#reward_data[i] is reward tokens
#the 0th 1st and 2nd index in the 2nd dimension are coordinates, active status, and value of reward


#python -m visdom.server -p 8097


#make punishment radius higher, possibly 2d Gaussian 
#possibly only consider the part of the image the track is on, so the tracker can work better
#test a neural network controlling with the video at the same time
#change punishments so they are all deactivated except the closest one

#update punishment update rewards, to get the reward

#glitch where the rover isn't getting positive reward sometimes,



#maybe give it a small negative reward when it chooses backwards, just to incentivize it to go forward instead, and it doesn't end up going around the track backwards
#tweak reward system, maybe positive rewards are inverely proportional to distance away 




#change where image is processed to actively store state,action,reward next_state




def click_rewards(filename,r_value=3,p_value=-1):
    '''starts interactive reward clicking, and stores file, returns reward_tokens,punish_tokens'''
    #pdb.set_trace()
    tracker=rover_tracker(reward_tracker,punish_tracker)
    tracker.get_frame()
    track_img=tracker.frame[:,:,::-1]
    plt.imshow(track_img)
    r=Interactive_rewards(track_img)
    wait = input("PRESS ENTER TO CONTINUE.")
    return create_n_store_rewards(r.rewards,r.punishments,filename,r_value,p_value)


def create_n_store_rewards(rewards,punishments,filename,r_value,p_value):
    '''Stores a list of two dictionaries to a file'''
    n_of_rewards=len(rewards)
    reward_tokens={'coord':np.array(rewards,dtype=np.int),'active':np.ones((n_of_rewards,1),dtype=np.int),'value':np.ones((n_of_rewards,1))*r_value}

    n_of_punishments=len(punishments)
    punish_tokens={'coord':np.array(punishments,dtype=np.int),'active':np.ones((n_of_punishments,1),dtype=np.int),'value':np.ones((n_of_punishments,1))*p_value}
    
    save_obj([reward_tokens,punish_tokens],filename)
    
    return reward_tokens,punish_tokens

def save_obj(obj, name ):
    with open('reward_tokens/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, 2)

def load_obj(name):
    with open( 'reward_tokens/'+name + '.pkl', 'rb') as f:
        return pickle.load(f)



class Reward_Tracker:
    def __init__(self,reward_tokens):
        self.reward_tokens=reward_tokens
        self.total=len(self.reward_tokens['coord'])
        self.last_update='all_forward'
        
    def update_rewards(self,rover_position):
        '''takes in lists of indexes to activate or de-activate'''
        activate=self.check_respawn(rover_position)
        deactivate,reward=self.find_touched_rewards(rover_position)
        np.put(self.reward_tokens['active'],activate,1)
        np.put(self.reward_tokens['active'],deactivate,0)
        
        closest=self.find_closest(rover_position)
        reward+=self.gaussian_2d(rover_position,self.reward_tokens['coord'][closest],magnitude=1000,sigma=15)*self.reward_tokens['active'][closest]
        
        c,closest_next=self.pass_reward_deactivate(rover_position,closest)
        
        
        print('closest = ' + str(closest))
        print('c is = '+ str(c))
        print('closest = ' + str(closest_next))
        
        
        if c > 0:
            np.put(self.reward_tokens['active'],closest,0)
        
        
        
        
        
        return reward
        
    def update_punishment(self,rover_position):
        '''updates for the punishments'''
        #pdb.set_trace()
        deactivate=np.arange(self.total)
        activate=self.find_closest(rover_position)
        np.put(self.reward_tokens['active'],deactivate,0) #deactive should be first here
        np.put(self.reward_tokens['active'],activate,1) #activates the closest one
        
        reward=self.gaussian_2d(rover_position,self.reward_tokens['coord'][activate])
        
        return reward

    def check_respawn(self,rover_position):
        '''finds the rewards to respawn'''
        add_rewards=[]
        x,y=rover_position
        #print(self.last_update)
        if self.last_update=='all_forward' and x>500 and y>345:
            add_rewards+=[20,21,22]
            self.last_update='last_few'
        
        if self.last_update=='last_few' and x < 266 and y > 345:
            add_rewards+=list(range(0,21))
            self.last_update='all_forward'
            
        return add_rewards
    
    def find_touched_rewards(self,rover_position,threshold=12):
        '''threshold is how many pixels away the rover's center needs to be from the reward to get it'''
        #find if the rover has touched any of the active rewards, returns reward indices
        distance=np.sqrt((np.square(rover_position-self.reward_tokens['coord']).sum(axis=1)))
        touched=np.where(distance<threshold+.1)
        
        #reward_value=(self.reward_tokens['value'][touched]).sum()
        reward_value=(self.reward_tokens['value'][touched]*self.reward_tokens['active'][touched]).sum()
        
        #if self.reward_tokens['active'][a].shape[0]>self.reward_tokens[]
        
        return touched,reward_value
    
    def find_closest(self,rover_position):
        index_max=np.sqrt((np.square(rover_position-self.reward_tokens['coord']).sum(axis=1))).argmin()
        
        return index_max
    
    def pass_reward_deactivate(self,rover_position,closest):
        '''returns the decision of deactivating a reward after passing it'''
        vectorA=rover_position-self.reward_tokens['coord'][closest]
        if closest!=self.total-1:
            closest_next=closest+1
        else:
            closest_next=0
            
        vectorL=self.reward_tokens['coord'][closest_next]-self.reward_tokens['coord'][closest]
        
        
        c=vectorA.dot(vectorL)/(np.sqrt((vectorL.dot(vectorL))))
        
        return c,closest_next
        

    def gaussian_2d(self,rover_position,punish_center,magnitude=300,sigma=10):
        x,y=rover_position
        a,b=punish_center
        output=(magnitude/sigma**2)*np.exp(-((x-a)**2+(y-b)**2)/(2*sigma**2))
        
        return output



class rover_tracker:
    def __init__(self,reward_tracker,punish_tracker):
        self.delta_thresh = 4
        self.min_area = 150                     # Threshold for triggering "motion detection"
        self.x,self.y,self.w,self.h=[0,0,0,0]
        self.current_reward=0
        #pdb.set_trace()
        self.cam = cv2.VideoCapture(0)             # Lets initialize capture on webcam
        self.avg = None
        self.reward_tracker=reward_tracker
        self.punish_tracker=punish_tracker
        self.rover_position=[0,0]
        self.accumulated_reward=0
    def get_frame(self):
        #pdb.set_trace()
        self.frame = self.cam.read()[1]
        
        #black out irrelevant part of image
        self.frame[:113,:,:]=0 #top down
        self.frame[:,:158,:]=0 #left to right
        self.frame[113:223,140:395,:]=0 #middle rectangle
    def track(self):
        #pdb.set_trace()
        grey = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        grey = cv2.GaussianBlur(grey, (21, 21), 0)
        
        # if the average frame is None, initialize it
        if self.avg is None:
            self.avg = grey.copy().astype("float")
            pass
        
        # accumulate the weighted average between the current frame and
        # previous frames, then compute the difference between the current
        # frame and running average
        cv2.accumulateWeighted(grey, self.avg, 0.5)
        frameDelta = cv2.absdiff(grey, cv2.convertScaleAbs(self.avg))

        # threshold the delta image, dilate the thresholded image to fill
        # in holes, then find contours on thresholded image
        thresh = cv2.threshold(frameDelta, self.delta_thresh, 255,
                               cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        #pdb.set_trace()
        # loop over the contours
        #if len(cnts) == 0:  # if no contours return None
        if cnts is None:
            print('cnts is None')
            return None
        if len(cnts) == 0:  # if no contours return None
            print('len(cnts)==0')
            return None
        else:
            for idx, c in enumerate(cnts):
                #pdb.set_trace()
                # if the contour is too small, ignore it
                
                #print(type(c));print(len(c));
                if cv2.contourArea(c) < self.min_area:
                    print('cv2.contourArea(c) < self.min_area')
                    return None

                # compute the bounding box for the contour, draw it on the frame,
                # and update the text
                (x, y, w, h) = cv2.boundingRect(c)
                
                # these if statements ensure that the tracker doesn't pick up
                # on any differences outside of the track.
                if x < 170:
                    print('x<170')
                    return None

                if y < 125 and x > 402:
                    print('y < 125 and x > 170')
                    return None
                elif y < 235 and x < 410:
                    print('y < 235 and x < 410')
                    return None

                # get the center of the bounding rectangle
                cx, cy = (x + w/2) // 2, (y + h/2) // 2
            #cv2.imshow('window', self.frame)
            #cv2.waitKey(1)
            #pdb.set_trace()
            #print(x,y,w,h)
            print('all the way')
            return [x,y,w,h]
        
    def show_video(self):
        #25.263 frames per second or a frame every 39.6 milliseconds
        
        while True:
            tracked=self.update()
            
            #cv2.startWindowThread()
            img=self.make_frame()
            
            plt.imshow(img)
            plt.pause(0.0001)
# =============================================================================
#             cv2.namedWindow("image")
#             cv2.imshow('image',img)
#             cv2.waitKey(1)     
# =============================================================================
            
            
            
            
# =============================================================================
#             tracker.get_frame();
#             center=tracker.track()
#             #cv2.imshow('window', tracker.frame)
#             font = cv2.FONT_HERSHEY_SIMPLEX
#             #pdb.set_trace()
#             for i,reward_coord in enumerate(self.reward_tracker.reward_tokens['coord']):
#                 
#                 if reward_tokens['active'][i]==1:
#                     cv2.circle(tracker.frame,tuple(reward_coord), reward_tracker.reward_tokens['value'][i], (0,255,0), -1)
#                     cv2.putText(tracker.frame,str(i),tuple(reward_coord+[-3,-10]), font, .3, (0,0,0), 1, cv2.LINE_AA)
#             #cv2.circle(tracker.frame,(0,285), 10, (0,255,0), -1)
#             
#             cv2.imshow('image',tracker.frame)
#             cv2.waitKey(1)     
# =============================================================================
    def show_visdom_video(self):
            
        while True:
            
            img=self.make_frame()
            img = img[:,:,::-1]
            vis.image(img.transpose((2,0,1)),win='image')
        
    
    def update(self):
        #pdb.set_trace()
        self.get_frame();
        tracked=self.track()
        if tracked is None:
            pass;#self.x,self.y,self.w,self.h=[0,0,0,0]
        else:
            self.x,self.y,self.w,self.h=tracked
        
        self.rover_position=[self.x+self.w//2,self.y+self.h//2]
        
        positive_reward=self.reward_tracker.update_rewards(rover_position=self.rover_position)
        negative_reward=self.punish_tracker.update_punishment(rover_position=self.rover_position)
        
        #self.punish_tracker.update_rewards(rover_position=self.rover_position)
        
        return tracked,positive_reward,negative_reward

    def make_frame(self):
        
        tracked,positive_reward,negative_reward=self.update()
# =============================================================================
#         if tracked is None:
#             x,y,w,h=[50,50,5,5]
#         else:
#             x,y,w,h=tracked
# =============================================================================
            
        #pdb.set_trace()
            
        #cv2.imshow('window', tracker.frame)
        font = cv2.FONT_HERSHEY_SIMPLEX
        #pdb.set_trace()
        for i,reward_coord in enumerate(self.reward_tracker.reward_tokens['coord']):
            
            if self.reward_tracker.reward_tokens['active'][i]==1:
                cv2.circle(self.frame,tuple(reward_coord), self.reward_tracker.reward_tokens['value'][i], (0,255,0), -1)
                cv2.putText(self.frame,str(i),tuple(reward_coord+[-3,-10]), font, .3, (0,0,0), 1, cv2.LINE_AA)
                
        
        for i,reward_coord in enumerate(self.punish_tracker.reward_tokens['coord']):
        
            if self.punish_tracker.reward_tokens['active'][i]==1:
                #pdb.set_trace()
                cv2.circle(self.frame,tuple(reward_coord), 20*abs(self.punish_tracker.reward_tokens['value'][i]), (0,0,255), 1)
                cv2.circle(self.frame,tuple(reward_coord), 15*abs(self.punish_tracker.reward_tokens['value'][i]), (0,0,255), 1)
                cv2.circle(self.frame,tuple(reward_coord), 10*abs(self.punish_tracker.reward_tokens['value'][i]), (0,0,255), 1)
                cv2.circle(self.frame,tuple(reward_coord), 5*abs(self.punish_tracker.reward_tokens['value'][i]), (0,0,255), -1)
                cv2.putText(self.frame,str(i),tuple(reward_coord+[-3,-10]), font, .3, (0,0,0), 1, cv2.LINE_AA)
        
        #draw bounding box as well
        #print(x,y,w,h)
        cv2.rectangle(self.frame,(self.x,self.y),(self.x+self.w,self.y+self.h),(0,255,0),2)
        cv2.circle(self.frame,(self.x+self.w//2,self.y+self.h//2),3,(0,0,0),-1)
        #pdb.set_trace()
        #Draw rewards
        cv2.putText(self.frame,'+ = '+str(round(positive_reward,3)),tuple([50,100]), font, 1, (50,255,50), 1, cv2.LINE_AA)
        cv2.putText(self.frame,'- = '+str(round(negative_reward,3)),tuple([50,130]), font, 1, (0,0,255), 1, cv2.LINE_AA)
        
        self.net_reward=positive_reward-negative_reward
        self.accumulated_reward+=self.net_reward
        if self.net_reward>0:
            clr=(50,255,50)

        else:
            clr=(0,0,255)
        if self.accumulated_reward>0:
            acc_clr=(50,255,50)
        else:
            acc_clr=(0,0,255)
        
        cv2.putText(self.frame,'net = '+str(round(self.net_reward,3)),tuple([8,160]), font, 1, clr, 1, cv2.LINE_AA)
        cv2.putText(self.frame,'accumulated_reward = '+str(round(self.accumulated_reward,3)),tuple([325,75]), font, .5, acc_clr, 1, cv2.LINE_AA)
        
        
# =============================================================================
#         cv2.putText(self.frame,'active = '+np.array2string(np.arange(22)),tuple([5,20]), font, .3, acc_clr, 1, cv2.LINE_AA)
#         cv2.putText(self.frame,'active = '+np.array2string(self.reward_tracker.reward_tokens['active'][:,0]),tuple([5,30]), font, .3, acc_clr, 1, cv2.LINE_AA)
#         cv2.putText(self.frame,'value = '+np.array2string(self.reward_tracker.reward_tokens['value'][:,0]),tuple([5,40]), font, .3, acc_clr, 1, cv2.LINE_AA)
# =============================================================================
        
        #cv2.circle(tracker.frame,(0,285), 10, (0,255,0), -1)
        
        #cv2.imshow('image',tracker.frame)
        #cv2.waitKey(1)     
        
        return self.frame

# =============================================================================
# reward_tracker=Reward_Tracker(reward_tokens)
# punish_tracker=Reward_Tracker(punish_tokens)
# =============================================================================
        
def see_map():
    tracker=rover_tracker(reward_tracker,punish_tracker)
    tracker.get_frame()
    plt.imshow(tracker.frame[:,:,::-1])
    
reward_tokens,punish_tokens=load_obj('rewards_0_prot2')

reward_tracker=Reward_Tracker(reward_tokens)
reward_tracker.reward_tokens['active'][-3:]=0
punish_tracker=Reward_Tracker(punish_tokens)

vis=visdom.Visdom()
if __name__ == "j":


    #=Reward_Tracker(reward_tokens)
    #pdb.set_trace()
    
    
# =============================================================================
#     reward_tokens,punish_tokens=load_obj('reward_tokens_0')
#     
#     reward_tracker=Reward_Tracker(reward_tokens)
#     punish_tracker=Reward_Tracker(punish_tokens)
# =============================================================================
    
    
    tracker = rover_tracker(reward_tracker,punish_tracker)
    tracker.get_frame()
    
    
    
    tracker.show_visdom_video()
    
    
    
    
    
    #tracker.show_video()
# =============================================================================
# while True:
#     tracker.get_frame()
#     #pdb.set_trace()
#     center = tracker.track()
#     
#     cv2.imshow('window', tracker.frame)
#     cv2.waitKey(1)
# =============================================================================
    

