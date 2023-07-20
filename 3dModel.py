import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import math

vertices = (
    (-0.5, -0.5, 0.5),
    (0.5, -0.5, 0.5),
    (0.5, 0.5, 0.5),
    (-0.5, 0.5, 0.5),
    (-0.5, -0.5, -0.5),
    (0.5, -0.5, -0.5),
    (0.5, 0.5, -0.5),
    (-0.5, 0.5, -0.5)
)

COLOR_RED = [1, 0, 0]
COLOR_GREEN = [0, 1, 0]
COLOR_BLUE = [0, 0, 1]
COLOR_YELLOW = [1, 1, 0]
COLOR_PURPLE = [1, 0, 1]
COLOR_CYAN = [0, 1, 1]

color_array = [
    [[COLOR_RED, COLOR_RED], [COLOR_RED, COLOR_RED]],
    [[COLOR_GREEN, COLOR_GREEN], [COLOR_GREEN, COLOR_GREEN]],
    [[COLOR_BLUE, COLOR_BLUE], [COLOR_BLUE, COLOR_BLUE]],
    [[COLOR_YELLOW, COLOR_YELLOW], [COLOR_YELLOW, COLOR_YELLOW]],
    [[COLOR_PURPLE, COLOR_PURPLE], [COLOR_PURPLE, COLOR_PURPLE]],
    [[COLOR_CYAN, COLOR_CYAN], [COLOR_CYAN, COLOR_CYAN]],
]

edges = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4)
)

class SmallCube:
    def __init__(self, position, colors):
        self.position = position
        self.colors = colors

    def draw(self):
        glPushMatrix()
        glTranslatef(self.position[0], self.position[1], self.position[2])
        for i in range(12):
            glBegin(GL_LINES)
            for vertex in edges[i]:
                glVertex3fv(vertices[vertex])
            glEnd()
        glPopMatrix()

pygame.init()
display = (800, 600)
pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

gluPerspective(115, (display[0] / display[1]), 0.1, 50.0)

glTranslatef(0.0, 0.0, -5)

rotation_enabled = False
rotation_start = (0, 0)

def draw_cube():
    for x in range(-1, 2):
        for y in range(-1, 2):
            for z in range(-1, 2):
                position = [x, y, z]
                colors=[]
                cube = SmallCube(position, colors)
                cube.draw()

while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: 
                    rotation_enabled = True
                    rotation_start = pygame.mouse.get_pos()
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1: 
                    rotation_enabled = False
            elif event.type == pygame.MOUSEMOTION and rotation_enabled:
                # Handle mouse rotation
                current_pos = pygame.mouse.get_pos()
                delta_x = current_pos[0] - rotation_start[0]
                delta_y = current_pos[1] - rotation_start[1]
                rotation_start = current_pos
                glRotatef(delta_x, 0, 1, 0)
                glRotatef(delta_y, 1, 0, 0)
        
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        draw_cube()
        pygame.display.flip()
        pygame.time.wait(10)
