import pygame
from OpenGL.GL import *
from OpenGL.GLU import *
import math
from constants import (
    SHAPES, SHAPE_COLORS,
    CUBE_VERTICES, PYRAMID_VERTICES, RECTANGLE_VERTICES,
    CUBE_EDGES, PYRAMID_EDGES, RECTANGLE_EDGES,
    CUBE_SURFACES, PYRAMID_FACES, RECTANGLE_SURFACES
)

def get_shape_height(shape_type):
    if shape_type == 'cube':
        return 1.0
    elif shape_type == 'sphere':
        return 0.8
    elif shape_type == 'pyramid':
        return 1.1
    elif shape_type == 'rectangle':
        return 0.6
    return 1.0

def draw_cube(pos, colors, highlight=False, angle=0):
    glPushMatrix()
    glTranslatef(*pos)
    glRotatef(angle, 0, 0, 1)
    glBegin(GL_QUADS)
    for i, surface in enumerate(CUBE_SURFACES):
        glColor3fv(colors[i % len(colors)])
        for vertex in surface:
            glVertex3fv(CUBE_VERTICES[vertex])
    glEnd()
    if highlight:
        glColor3fv((1,0,0))
        glLineWidth(4)
        glBegin(GL_LINES)
        for edge in CUBE_EDGES:
            for vertex in edge:
                glVertex3fv(CUBE_VERTICES[vertex])
        glEnd()
        glLineWidth(1)
    glPopMatrix()

def draw_sphere(pos, colors, highlight=False, angle=0):
    glPushMatrix()
    glTranslatef(*pos)
    glRotatef(angle, 0, 0, 1)
    quad = gluNewQuadric()
    for i, color in enumerate(colors):
        glColor3fv(color)
        gluSphere(quad, 0.8, 32, 32)
    if highlight:
        glColor3fv((1,0,0))
        glLineWidth(4)
        gluQuadricDrawStyle(quad, GLU_LINE)
        gluSphere(quad, 0.82, 16, 16)
        glLineWidth(1)
    glPopMatrix()

def draw_pyramid(pos, colors, highlight=False, angle=0):
    glPushMatrix()
    glTranslatef(*pos)
    glRotatef(angle, 0, 0, 1)
    glBegin(GL_TRIANGLES)
    for i, face in enumerate(PYRAMID_FACES):
        glColor3fv(colors[i % len(colors)])
        for vertex in face:
            glVertex3fv(PYRAMID_VERTICES[vertex])
    glEnd()
    if highlight:
        glColor3fv((1,0,0))
        glLineWidth(4)
        glBegin(GL_LINES)
        for edge in PYRAMID_EDGES:
            for vertex in edge:
                glVertex3fv(PYRAMID_VERTICES[vertex])
        glEnd()
        glLineWidth(1)
    glPopMatrix()

def draw_rectangle(pos, colors, highlight=False, angle=0):
    glPushMatrix()
    glTranslatef(*pos)
    glRotatef(angle, 0, 0, 1)
    glBegin(GL_QUADS)
    for i, surface in enumerate(RECTANGLE_SURFACES):
        glColor3fv(colors[i % len(colors)])
        for vertex in surface:
            glVertex3fv(RECTANGLE_VERTICES[vertex])
    glEnd()
    if highlight:
        glColor3fv((1,0,0))
        glLineWidth(4)
        glBegin(GL_LINES)
        for edge in RECTANGLE_EDGES:
            for vertex in edge:
                glVertex3fv(RECTANGLE_VERTICES[vertex])
        glEnd()
        glLineWidth(1)
    glPopMatrix()

class Shape3D:
    def __init__(self, pos, shape_type):
        self.pos = list(pos)
        self.shape_type = shape_type
        self.colors = SHAPE_COLORS[shape_type]
        self.picked = False
        self.angle = 0
        self.falling = False
        self.velocity = 0.0
        self.size = get_shape_height(shape_type)
        self.vel_x = 0.0
        self.vel_z = 0.0
        self.shake_frames = 0
        self.shake_angle = 0.0
        self.state = 'idle'  # idle, picked, falling, deleted

    def draw(self, highlight=False, ghost=False):
        if ghost:
            glPushAttrib(GL_ENABLE_BIT)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glColor4f(0.7, 0.7, 0.7, 0.4)
        # Hiệu ứng rung lắc
        angle = self.angle
        if self.shake_frames > 0:
            angle += math.sin(self.shake_frames * 0.5) * self.shake_angle
        if self.shape_type == 'cube':
            draw_cube(self.pos, self.colors, highlight, angle)
        elif self.shape_type == 'sphere':
            draw_sphere(self.pos, self.colors, highlight, angle)
        elif self.shape_type == 'pyramid':
            draw_pyramid(self.pos, self.colors, highlight, angle)
        elif self.shape_type == 'rectangle':
            draw_rectangle(self.pos, self.colors, highlight, angle)
        if ghost:
            glDisable(GL_BLEND)
            glPopAttrib() 