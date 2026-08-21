import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import cv2
import mediapipe as mp
from hand_tracker import HandTracker
from mannequin import Mannequin
from constants import (
    DISPLAY_SIZE, FPS, GRAVITY,
    CAMERA_DISTANCE, CAMERA_FOV, CAMERA_NEAR, CAMERA_FAR
)
import time

class Game:
    def __init__(self):
        # Khởi tạo pygame
        pygame.init()
        pygame.display.set_mode(DISPLAY_SIZE, DOUBLEBUF | OPENGL)
        pygame.display.set_caption("3D Mannequin Control")

        # Khởi tạo OpenGL
        self.init_opengl()

        # Khởi tạo hand tracker
        self.hand_tracker = HandTracker()
        self.hand_tracker.start()

        # Khởi tạo hình nộm
        self.mannequin = Mannequin([0, 0, 0])

        # Biến để theo dõi trạng thái tương tác
        self.is_grabbing = False
        self.grabbed_joint = None
        self.last_hand_pos = None

    def init_opengl(self):
        """Khởi tạo OpenGL"""
        # Thiết lập viewport
        glViewport(0, 0, DISPLAY_SIZE[0], DISPLAY_SIZE[1])
        
        # Thiết lập ma trận chiếu
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(CAMERA_FOV, (DISPLAY_SIZE[0]/DISPLAY_SIZE[1]), CAMERA_NEAR, CAMERA_FAR)
        
        # Thiết lập ma trận modelview
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glTranslatef(0.0, 0.0, -CAMERA_DISTANCE)
        
        # Bật các tính năng
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        
        # Thiết lập ánh sáng
        glLightfv(GL_LIGHT0, GL_POSITION, (0, 10, 0, 1))
        glLightfv(GL_LIGHT0, GL_AMBIENT, (0.2, 0.2, 0.2, 1))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.8, 0.8, 0.8, 1))

    def handle_hand_interaction(self, hand_landmarks):
        """Xử lý tương tác với bàn tay"""
        if hand_landmarks is None:
            if self.is_grabbing:
                # Thả khớp nếu không còn phát hiện tay
                self.mannequin.release_joint(self.grabbed_joint)
                self.is_grabbing = False
                self.grabbed_joint = None
            return
        
        # Lấy vị trí của bàn tay (sử dụng điểm giữa lòng bàn tay)
        palm_pos = np.array([
            hand_landmarks.landmark[0].x * 2 - 1,  # Chuyển đổi từ [0,1] sang [-1,1]
            -(hand_landmarks.landmark[0].y * 2 - 1),
            hand_landmarks.landmark[0].z * 2
        ])
        
        # Kiểm tra cử chỉ nắm
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]
        distance = np.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)
        
        if not self.is_grabbing and distance < 0.1:  # Nếu ngón cái và ngón trỏ gần nhau
            # Tìm khớp gần nhất với vị trí bàn tay
            min_dist = float('inf')
            closest_joint = None
            
            for joint_name, joint in self.mannequin.joints.items():
                joint_pos = joint.get_world_position()
                dist = np.linalg.norm(joint_pos - palm_pos)
                if dist < min_dist and dist < 0.5:  # Chỉ nắm nếu trong phạm vi 0.5 đơn vị
                    min_dist = dist
                    closest_joint = joint_name
            
            if closest_joint:
                self.is_grabbing = True
                self.grabbed_joint = closest_joint
                self.mannequin.grab_joint(closest_joint, palm_pos)
                self.last_hand_pos = palm_pos
        
        elif self.is_grabbing:
            if distance >= 0.1:  # Nếu ngón cái và ngón trỏ xa nhau
                self.mannequin.release_joint(self.grabbed_joint)
                self.is_grabbing = False
                self.grabbed_joint = None
            else:
                # Di chuyển khớp theo bàn tay
                self.mannequin.move_grabbed_joint(palm_pos)
                self.last_hand_pos = palm_pos

    def run(self):
        """Chạy game loop"""
        clock = pygame.time.Clock()
        last_time = time.time()
        
        while True:
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        return
            
            # Xử lý tương tác với tay
            hand_landmarks = self.hand_tracker.get_hand_landmarks()
            self.handle_hand_interaction(hand_landmarks)
            
            # Cập nhật vật lý
            gravity = np.array([0, GRAVITY, 0])
            self.mannequin.update(dt, gravity)
            
            # Xóa màn hình
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            
            # Reset ma trận modelview
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()
            glTranslatef(0.0, 0.0, -CAMERA_DISTANCE)
            
            # Vẽ hình nộm
            self.mannequin.draw()
            
            # Vẽ lưới tham chiếu
            glBegin(GL_LINES)
            glColor3f(0.5, 0.5, 0.5)
            for i in range(-5, 6):
                glVertex3f(i, 0, -5)
                glVertex3f(i, 0, 5)
                glVertex3f(-5, 0, i)
                glVertex3f(5, 0, i)
            glEnd()
            
            pygame.display.flip()
            clock.tick(FPS)

if __name__ == "__main__":
    game = Game()
    game.run()