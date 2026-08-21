import pygame
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import os

def load_textures():
    textures = glGenTextures(6)
    for i in range(6):
        fname = f"face{i}.png"
        if os.path.exists(fname):
            img = pygame.image.load(fname)
        else:
            # Tạo surface màu nếu không có ảnh
            img = pygame.Surface((64,64))
            color = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255)][i]
            img.fill(color)
        img = pygame.transform.flip(img, False, True)
        img_data = pygame.image.tostring(img, "RGBA", True)
        glBindTexture(GL_TEXTURE_2D, textures[i])
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img.get_width(), img.get_height(), 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)
    return textures

def draw_text(screen, text, pos, color=(255,0,0)):
    font = pygame.font.Font(None, 40)  # Font mặc định, size lớn
    text_surface = font.render(text, True, color)
    screen.blit(text_surface, pos)

def draw_cursor(pos):
    glPushMatrix()
    glDisable(GL_TEXTURE_2D)
    glTranslatef(*pos)
    glColor3fv((1,0,0))
    quad = gluNewQuadric()
    gluSphere(quad, 0.4, 16, 16)
    glPopMatrix()
    # Vẽ bóng con trỏ trên mặt đất
    glPushMatrix()
    glTranslatef(pos[0], -2.0, pos[2])
    glColor4f(0.2,0.2,0.2,0.3)
    quad2 = gluNewQuadric()
    gluSphere(quad2, 0.18, 12, 12)
    glPopMatrix()

def draw_dashed_line(start, end, dash_length=1.0, gap_length=1.0):
    start = np.array(start)
    end = np.array(end)
    vec = end - start
    length = np.linalg.norm(vec)
    if length == 0:
        return
    direction = vec / length
    num_dashes = int(length // (dash_length + gap_length))
    for i in range(num_dashes + 1):
        p1 = start + direction * (i * (dash_length + gap_length))
        p2 = start + direction * (i * (dash_length + gap_length) + dash_length)
        if np.linalg.norm(p2 - start) > length:
            p2 = end
        glBegin(GL_LINES)
        glVertex3fv(p1)
        glVertex3fv(p2)
        glEnd()

def draw_bounding_box(min_x, max_x, min_y, max_y, min_z, max_z):
    # Các cạnh nét liền (trước và cạnh chính)
    solid_edges = [
        # bottom face
        ((min_x, min_y, min_z), (max_x, min_y, min_z)),
        ((max_x, min_y, min_z), (max_x, min_y, max_z)),
        ((max_x, min_y, max_z), (min_x, min_y, max_z)),
        ((min_x, min_y, max_z), (min_x, min_y, min_z)),
        # vertical front
        ((min_x, min_y, min_z), (min_x, max_y, min_z)),
        ((max_x, min_y, min_z), (max_x, max_y, min_z)),
        ((max_x, min_y, max_z), (max_x, max_y, max_z)),
        ((min_x, min_y, max_z), (min_x, max_y, max_z)),
        # top face
        ((min_x, max_y, min_z), (max_x, max_y, min_z)),
        ((max_x, max_y, min_z), (max_x, max_y, max_z)),
        ((max_x, max_y, max_z), (min_x, max_y, max_z)),
        ((min_x, max_y, max_z), (min_x, max_y, min_z)),
    ]
    # Các cạnh nét đứt (cạnh khuất)
    dashed_edges = [
        # 4 đường chéo phía sau
        ((min_x, min_y, min_z), (max_x, max_y, max_z)),
        ((max_x, min_y, min_z), (min_x, max_y, max_z)),
        ((min_x, min_y, max_z), (max_x, max_y, min_z)),
        ((max_x, min_y, max_z), (min_x, max_y, min_z)),
    ]
    # Vẽ nét liền
    glColor3f(0,0,0)
    glLineWidth(2)
    glBegin(GL_LINES)
    for start, end in solid_edges:
        glVertex3fv(start)
        glVertex3fv(end)
    glEnd()
    # Vẽ nét đứt
    glLineWidth(1)
    for start, end in dashed_edges:
        draw_dashed_line(start, end, dash_length=1.0, gap_length=1.0)

def draw_ground(min_x, max_x, min_z, max_z, y):
    glColor3f(1,1,1)
    glBegin(GL_QUADS)
    glVertex3f(min_x, y, min_z)
    glVertex3f(max_x, y, min_z)
    glVertex3f(max_x, y, max_z)
    glVertex3f(min_x, y, max_z)
    glEnd()