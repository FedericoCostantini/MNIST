import pygame as pg
from pygame.locals import *
import sys
import NNv2, NNv1
import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt

scale: int = 5
WIDTH: int = 28 * scale
HEIGHT: int = 28 * scale
FPS: int = 120
BLACK: tuple = (0, 0, 0)
WHITE: tuple = (255, 255, 255)
RADIUS: int = 5

class DigitVisualizer:
    def __init__(self) -> None:
        pg.init()
        pg.mixer.init()
        self.screen = pg.display.set_mode((WIDTH, HEIGHT))
        pg.display.set_caption("MNIST")
        self.clock = pg.time.Clock()
        self.surface = pg.display.get_surface()
        self.screen.fill(BLACK)
        try:
            self.nn = NNv2.NeuralNetwork(True)
        except:
            self.nn = NNv2.NeuralNetwork(False)
            self.nn.train()
        print("Current precision of the network: {0}".format(self.nn.evaluate()))
        
    def convert_screen(self) -> np.ndarray:
        original: np.ndarray = np.zeros((WIDTH, HEIGHT))
        
        for i in range(WIDTH):
            for j in range(HEIGHT):
                original[i, j] = self.screen.get_at((j, i))[0]
        
        reshaped_image = resize(original, (28, 28), anti_aliasing=True).astype(np.float64) / 255
        
        '''
        # Visual checker of the data
        plt.imshow(reshaped_image, cmap='gray')
        plt.axis('off')
        plt.show()
        '''
        
        return reshaped_image
    
    def game_loop(self) -> None:
        while True:
            self.clock.tick(FPS)

            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()
                    sys.exit()
                if event.type == pg.KEYDOWN:
                    if event.key == pg.K_a:
                        self.screen.fill(BLACK)
                    elif event.key == pg.K_s:
                        self.nn.activate(self.convert_screen())
            
            if pg.mouse.get_pressed(num_buttons=3)[0] == True:
                pg.draw.circle(self.surface, WHITE, pg.mouse.get_pos(), RADIUS)
                
            pg.display.flip()



mnist = DigitVisualizer()
mnist.game_loop()