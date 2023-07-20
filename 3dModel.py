import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import math

vertices = [
    # Front Face
    [0.5, -0.5, 0.5],
    [0.5, 0.5, 0.5],
    [-0.5, 0.5, 0.5],
    [-0.5, -0.5, 0.5],
    # Back Face
    [0.5, -0.5, -0.5],
    [0.5, 0.5, -0.5],
    [-0.5, 0.5, -0.5],
    [-0.5, -0.5, -0.5],
    # Left Face
    [-0.5, -0.5, 0.5],
    [-0.5, 0.5, 0.5],
    [-0.5, 0.5, -0.5],
    [-0.5, -0.5, -0.5],
    # Right Face
    [0.5, -0.5, 0.5],
    [0.5, 0.5, 0.5],
    [0.5, 0.5, -0.5],
    [0.5, -0.5, -0.5],
    # Top Face
    [0.5, 0.5, 0.5],
    [0.5, 0.5, -0.5],
    [-0.5, 0.5, -0.5],
    [-0.5, 0.5, 0.5],
    # Bottom Face
    [0.5, -0.5, 0.5],
    [0.5, -0.5, -0.5],
    [-0.5, -0.5, -0.5],
    [-0.5, -0.5, 0.5],
]

vertice = [
    (-0.5, -0.5, 0.5),
    (0.5, -0.5, 0.5),
    (0.5, 0.5, 0.5),
    (-0.5, 0.5, 0.5),
    (-0.5, -0.5, -0.5),
    (0.5, -0.5, -0.5),
    (0.5, 0.5, -0.5),
    (-0.5, 0.5, -0.5)
]

edge = [
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
]

edges = [
    (0, 1, 2, 3),    # Front Face
    (4, 5, 6, 7),    # Back Face
    (8, 9, 10, 11),  # Left Face
    (12, 13, 14, 15),  # Right Face
    (16, 17, 18, 19),  # Top Face
    (20, 21, 22, 23),  # Bottom Face
]



edge = (
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

    def draw_cube(self):
        glPushMatrix()
        glTranslatef(self.position[0], self.position[1], self.position[2])
        for i in range(6):
            glBegin(GL_QUADS)
            for vertex in edges[i]:
                glColor3fv(self.colors[i])
                #glColor3f(1,1,1)
                glVertex3fv(vertices[vertex])
            glEnd()
        glPopMatrix()

    def draw_border(self):
        glPushMatrix()
        glTranslatef(self.position[0], self.position[1], self.position[2])
        for i in range(12):
            #glEnable(GL_LINE_SMOOTH)  # Enable line antialiasing
            glLineWidth(5.0)
            glBegin(GL_LINES)
            for vertex in edge[i]:
                glColor3f(0,0,0)
                glVertex3fv(vertice[vertex])
            glEnd()
        glPopMatrix()

pygame.init()
display = (800, 600)
pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

glEnable(GL_DEPTH_TEST)
gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)

glTranslatef(0.0, 0.0, -15)

rotation_enabled = False
rotation_start = (0, 0)

def draw_cube():
    for x in range(-1, 2):
        for y in range(-1, 2):
            for z in range(-1, 2):
                position = [x, y, z]
                colors = [(0,1,0), (0,0,1), (1,0.5,0), (1,0,0), (1,1,1), (1,1,0)]
                cube = SmallCube(position, colors)
                cube.draw_cube()
                cube.draw_border()

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
