import pygame
from pygame.locals import *
import pygame_gui
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import math
from itertools import zip_longest

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

    def updatePosition(self, position):
        self.position = position

    def draw_cube(self):
        glPushMatrix()
        glTranslatef(self.position[0], self.position[1], self.position[2])
        for i in range(6):
            glBegin(GL_QUADS)
            for vertex in edges[i]:
                glColor3fv(self.colors[i])
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



def draw_cube(CustomPosition):
    for x in range(-1, 2):
        for y in range(-1, 2):
            for z in range(-1, 2):
                position = [x, y, z]
                if (position == CustomPosition):
                    colors = [(0,1,0), (0,0,1), (1,0.5,0), (1,0,0), (1,1,1), (1,1,0)]
                else:
                    colors = [(1,1,1)]*6
                cube = SmallCube(position, colors)
                cube.draw_cube()
                cube.draw_border()



#draw_cube([-1,-1,1])


cube_positions = [[-1, -1, 1], [0, -1, 1], [1, -1, 1],
    [-1, 0, 1], [0, 0, 1], [1, 0, 1],
    [-1, 1, 1], [0, 1, 1], [1, 1, 1],
    [-1, -1, 0], [0, -1, 0], [1, -1, 0],
    [-1, 0, 0], [0, 0, 0], [1, 0, 0],
    [-1, 1, 0], [0, 1, 0], [1, 1, 0],
    [-1, -1, -1], [0, -1, -1], [1, -1, -1],
    [-1, 0, -1], [0, 0, -1], [1, 0, -1],
    [-1, 1, -1], [0, 1, -1], [1, 1, -1]]

cubes = []
colors = []


for pos in cube_positions:
    colors = [(0,1,0), (0,0,1), (1,0.5,0), (1,0,0), (1,1,1), (1,1,0)]
    cubes.append(SmallCube(pos,colors))





def rotate_front():
    new_colors = [cube.colors[:] for cube in cubes]
    new_colors[0] = [(0,1,0), (0,0,1), new_colors[0][5], new_colors[0][4], new_colors[0][2], new_colors[0][3]]
    new_colors[1] = [(0,1,0), (0,0,1), new_colors[1][5], new_colors[1][4], new_colors[1][2], new_colors[1][3]]
    new_colors[2] = [(0,1,0), (0,0,1), new_colors[2][5], new_colors[2][4], new_colors[2][2], new_colors[2][3]]
    new_colors[3] = [(0,1,0), (0,0,1), new_colors[3][5], new_colors[3][4], new_colors[3][2], new_colors[3][3]]
    new_colors[4] = [(0,1,0), (0,0,1), new_colors[4][5], new_colors[4][4], new_colors[4][2], new_colors[4][3]]
    new_colors[5] = [(0,1,0), (0,0,1), new_colors[5][5], new_colors[5][4], new_colors[5][2], new_colors[5][3]]
    new_colors[6] = [(0,1,0), (0,0,1), new_colors[6][5], new_colors[6][4], new_colors[6][2], new_colors[6][3]]
    new_colors[7] = [(0,1,0), (0,0,1), new_colors[7][5], new_colors[7][4], new_colors[7][2], new_colors[7][3]]
    new_colors[8] = [(0,1,0), (0,0,1), new_colors[8][5], new_colors[8][4], new_colors[8][2], new_colors[8][3]]

    for i in range(9):
        cubes[i].colors = new_colors[i]


middle = [(1,1,1), (1,0,0), (1,1,0), (1,0.5,0)]
def rotate_middle():
    last_color = middle[-1]
    middle.pop()
    middle.insert(0, last_color)
    new_colors = [cube.colors[:] for cube in cubes]
    new_colors[9] = [(0,1,0), (0,0,1), middle[3], middle[1], middle[0], middle[2]]
    new_colors[10] = [(0,1,0), (0,0,1), middle[3], middle[1], middle[0], middle[2]]
    new_colors[11] = [(0,1,0), (0,0,1), middle[3], middle[1], middle[0], middle[2]]
    new_colors[12] = [(0,1,0), (0,0,1), middle[3], middle[1], middle[0], middle[2]]
    new_colors[13] = [(0,1,0), (0,0,1), middle[3], middle[1], middle[0], middle[2]]
    new_colors[14] = [(0,1,0), (0,0,1), middle[3], middle[1], middle[0], middle[2]]
    new_colors[15] = [(0,1,0), (0,0,1), middle[3], middle[1], middle[0], middle[2]]
    new_colors[16] = [(0,1,0), (0,0,1), middle[3], middle[1], middle[0], middle[2]]
    new_colors[17] = [(0,1,0), (0,0,1), middle[3], middle[1], middle[0], middle[2]]

    for i in range(9,19):
        cubes[i].colors = new_colors[i]



k = 1
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
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_f:
                    rotate_front()
                if event.key == pygame.K_m:
                    rotate_middle()
            elif event.type == pygame.MOUSEMOTION and rotation_enabled:
                # Handle mouse rotation
                current_pos = pygame.mouse.get_pos()
                delta_x = current_pos[0] - rotation_start[0]
                delta_y = current_pos[1] - rotation_start[1]
                rotation_start = current_pos
                glRotatef(delta_x, 0, 1, 0)
                glRotatef(delta_y, 1, 0, 0)

        
        #glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        for cube in cubes:
            glPushMatrix()
            cube.draw_cube()
            cube.draw_border()
            glPopMatrix()
        pygame.display.flip()
        pygame.time.wait(10)


        
