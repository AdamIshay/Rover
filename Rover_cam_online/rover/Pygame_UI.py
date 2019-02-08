import pygame
import numpy as np
import cv2
from scipy.misc import bytescale
import os
import time

class Pygame_UI:
    def __init__(self, fps, speed):
        pygame.init()
        pygame.display.set_caption('Rover Dashboard')
        self.screen_size = [700, 480]
        self.screen = pygame.display.set_mode(self.screen_size)
        self.screen.fill((255,255,255))
        self.fontSize = 30
        self.font = pygame.font.SysFont(None, self.fontSize)
        self.clock = pygame.time.Clock()
        self.fps = fps
        self.start_time = time.time()
        self.color = (0,0,0)
        self.action_dict = {}
        self.action_dict['a'] = [-speed, speed, 0]
        self.action_dict[0] = [-speed, speed]
        self.action_dict['w'] = [speed, speed, 1]
        self.action_dict[1] = [speed, speed*1.4]
        self.action_dict['d'] = [speed, -speed, 2]
        self.action_dict[2] = [speed, -speed]
        self.action_dict['s'] = [-speed, -speed, 3]
        self.action_dict[3] = [-speed, -speed]
        self.action_dict[' '] = [0, 0, 4]
        self.action_dict[4] = [0, 0]
        self.action_dict['q'] = [0, 0, 9]


    def display_message(self, text, color, x, y):
        label = self.font.render(text, True, color)
        self.screen.blit(label, (x,y))

    def getActiveKey(self):
        key = None
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                key = event.key
                key = chr(key)
        return key

    def manage_UI(self):
        self.clock.tick(self.fps)
        os.system('clear')
        return

    def show_feed(self, image):
        cv2.imshow("RoverCam", bytescale(image))
        cv2.waitKey(1)
        return

    def cleanup(self):
        elapsed_time = np.round(time.time() - self.start_time, 2)
        print('This run lasted %.2f seconds'%(elapsed_time))
        pygame.quit()
        cv2.destroyAllWindows()
        return
