#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 16:29:05 2018

@author: mpcr
"""
import matplotlib.pyplot as plt
import numpy as np 
import pdb



class Interactive_rewards():
    def __init__(self,track_img):
        self.track_img=track_img
    
        
        self.fig,self.ax=plt.subplots()
        self.x_reward=[];self.y_reward=[];
        self.x_rover=[];self.y_rover=[];
        
        self.something=self.click_rewards()
        return 
    
    def onclick(self,event):
        if event.button == 1:
             self.x_reward.append(event.xdata)
             self.y_reward.append(event.ydata)
        
        if event.button == 3:
             self.x_rover.append(event.xdata)
             self.y_rover.append(event.ydata)
        if event.button == 2:
            event.canvas.mpl_disconnect(cid)
            self.rewards=np.array([self.something[0][0],self.something[0][1]]).T
            self.punishments=np.array([self.something[1][0],self.something[1][1]]).T
            return
        
        #clear frame
        plt.clf()
        plt.scatter(self.x_reward,self.y_reward,c=(0,1,0)); #inform matplotlib of the new data
        plt.scatter(self.x_rover,self.y_rover,c=(1,0,0));
        plt.imshow(self.track_img)
        
        plt.draw() #redraw
    
    def click_rewards(self):
    
        global cid
        plt.imshow(self.track_img)
        cid=self.fig.canvas.mpl_connect('button_press_event',self.onclick)
        plt.show()
        plt.draw()
        
        return [self.x_reward,self.y_reward],[self.x_rover,self.y_rover]
    





# =============================================================================
# 
# x_pts = []
# y_pts = []
# 
# fig, ax = plt.subplots()
# 
# line, = ax.plot(x_pts, y_pts, marker="o")
# 
# 
# plt.imshow(track)
# 
# 
# def onpick(event):
#     m_x, m_y = event.x, event.y
#     x, y = ax.transData.inverted().transform([m_x, m_y])
#     x_pts.append(x)
#     y_pts.append(y)
#     line.set_xdata(x_pts)
#     line.set_ydata(y_pts)
#     fig.canvas.draw()
# 
# 
# def getCoord(self):
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     plt.imshow(self.img)
#     cid = fig.canvas.mpl_connect('button_press_event', self.__onclick__)
#     plt.show()
#     return self.point
# 
# 
# 
# 
# fig.canvas.mpl_connect('button_press_event', onpick)
# 
# plt.show()
# =============================================================================


# =============================================================================
# The negative rewards will have a larger radius and will not disappear
# when received
# =============================================================================
