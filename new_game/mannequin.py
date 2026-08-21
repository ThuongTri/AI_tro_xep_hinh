import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
import math

class Joint:
    def __init__(self, pos, parent=None):
        self.pos = np.array(pos, dtype=np.float32)
        self.rotation = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # Euler angles
        self.parent = parent
        self.children = []
        self.length = 0.0
        self.radius = 0.1  # Bán kính khớp để vẽ
        self.velocity = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.angular_velocity = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.mass = 1.0
        self.is_grabbed = False
        self.grab_point = None

    def add_child(self, child):
        self.children.append(child)
        child.parent = self
        # Tính độ dài từ khớp hiện tại đến khớp con
        child.length = np.linalg.norm(child.pos - self.pos)

    def get_world_position(self):
        """Lấy vị trí của khớp trong không gian thế giới"""
        if self.parent is None:
            return self.pos
        # Áp dụng các phép biến đổi từ khớp cha
        world_pos = self.parent.get_world_position()
        # Áp dụng phép xoay
        rotation_matrix = self.get_rotation_matrix()
        rotated_pos = np.dot(rotation_matrix, self.pos - self.parent.pos)
        return world_pos + rotated_pos

    def get_rotation_matrix(self):
        """Tạo ma trận xoay từ các góc Euler"""
        rx, ry, rz = np.radians(self.rotation)
        
        # Ma trận xoay theo trục X
        Rx = np.array([
            [1, 0, 0],
            [0, math.cos(rx), -math.sin(rx)],
            [0, math.sin(rx), math.cos(rx)]
        ])
        
        # Ma trận xoay theo trục Y
        Ry = np.array([
            [math.cos(ry), 0, math.sin(ry)],
            [0, 1, 0],
            [-math.sin(ry), 0, math.cos(ry)]
        ])
        
        # Ma trận xoay theo trục Z
        Rz = np.array([
            [math.cos(rz), -math.sin(rz), 0],
            [math.sin(rz), math.cos(rz), 0],
            [0, 0, 1]
        ])
        
        # Kết hợp các ma trận xoay
        return np.dot(Rz, np.dot(Ry, Rx))

    def update_physics(self, dt, gravity):
        """Cập nhật vật lý cho khớp"""
        if not self.is_grabbed:
            # Áp dụng trọng lực
            self.velocity += gravity * dt
            
            # Cập nhật vị trí
            self.pos += self.velocity * dt
            
            # Áp dụng ma sát
            self.velocity *= 0.98
            self.angular_velocity *= 0.98

    def draw(self):
        """Vẽ khớp và các phần nối"""
        glPushMatrix()
        
        # Vẽ khớp
        glColor3f(0.8, 0.2, 0.2)  # Màu đỏ cho khớp
        quad = gluNewQuadric()
        gluSphere(quad, self.radius, 16, 16)
        
        # Vẽ các phần nối đến khớp con
        glColor3f(0.2, 0.2, 0.8)  # Màu xanh cho phần nối
        for child in self.children:
            # Tính vector hướng từ khớp hiện tại đến khớp con
            direction = child.pos - self.pos
            length = np.linalg.norm(direction)
            if length > 0:
                direction = direction / length
                
                # Tính góc xoay để căn chỉnh cylinder
                up = np.array([0, 1, 0])
                rotation_axis = np.cross(up, direction)
                rotation_angle = math.degrees(math.acos(np.dot(up, direction)))
                
                glPushMatrix()
                glRotatef(rotation_angle, *rotation_axis)
                cylinder = gluNewQuadric()
                gluCylinder(cylinder, 0.05, 0.05, length, 16, 1)
                glPopMatrix()
        
        glPopMatrix()
        
        # Vẽ đệ quy cho các khớp con
        for child in self.children:
            child.draw()

class Mannequin:
    def __init__(self, pos):
        self.pos = np.array(pos, dtype=np.float32)
        self.joints = {}
        self.setup_skeleton()
        self.grabbed_joint = None
        self.grab_point = None

    def setup_skeleton(self):
        """Thiết lập cấu trúc xương cho hình nộm"""
        # Tạo các khớp chính
        self.joints['spine'] = Joint([0, 0, 0])  # Cột sống
        self.joints['head'] = Joint([0, 1.7, 0])  # Đầu
        self.joints['left_shoulder'] = Joint([-0.3, 1.5, 0])  # Vai trái
        self.joints['right_shoulder'] = Joint([0.3, 1.5, 0])  # Vai phải
        self.joints['left_elbow'] = Joint([-0.6, 1.3, 0])  # Khuỷu tay trái
        self.joints['right_elbow'] = Joint([0.6, 1.3, 0])  # Khuỷu tay phải
        self.joints['left_hand'] = Joint([-0.9, 1.1, 0])  # Bàn tay trái
        self.joints['right_hand'] = Joint([0.9, 1.1, 0])  # Bàn tay phải
        self.joints['left_hip'] = Joint([-0.2, 0.9, 0])  # Hông trái
        self.joints['right_hip'] = Joint([0.2, 0.9, 0])  # Hông phải
        self.joints['left_knee'] = Joint([-0.2, 0.5, 0])  # Đầu gối trái
        self.joints['right_knee'] = Joint([0.2, 0.5, 0])  # Đầu gối phải
        self.joints['left_foot'] = Joint([-0.2, 0.1, 0])  # Bàn chân trái
        self.joints['right_foot'] = Joint([0.2, 0.1, 0])  # Bàn chân phải

        # Thiết lập cấu trúc phân cấp
        self.joints['spine'].add_child(self.joints['head'])
        self.joints['spine'].add_child(self.joints['left_shoulder'])
        self.joints['spine'].add_child(self.joints['right_shoulder'])
        self.joints['spine'].add_child(self.joints['left_hip'])
        self.joints['spine'].add_child(self.joints['right_hip'])
        
        self.joints['left_shoulder'].add_child(self.joints['left_elbow'])
        self.joints['right_shoulder'].add_child(self.joints['right_elbow'])
        
        self.joints['left_elbow'].add_child(self.joints['left_hand'])
        self.joints['right_elbow'].add_child(self.joints['right_hand'])
        
        self.joints['left_hip'].add_child(self.joints['left_knee'])
        self.joints['right_hip'].add_child(self.joints['right_knee'])
        
        self.joints['left_knee'].add_child(self.joints['left_foot'])
        self.joints['right_knee'].add_child(self.joints['right_foot'])

    def update(self, dt, gravity):
        """Cập nhật trạng thái của hình nộm"""
        for joint in self.joints.values():
            joint.update_physics(dt, gravity)

    def draw(self):
        """Vẽ hình nộm"""
        glPushMatrix()
        glTranslatef(*self.pos)
        
        # Vẽ tất cả các khớp
        self.joints['spine'].draw()
        
        glPopMatrix()

    def grab_joint(self, joint_name, grab_point):
        """Nắm lấy một khớp"""
        if joint_name in self.joints:
            self.joints[joint_name].is_grabbed = True
            self.joints[joint_name].grab_point = grab_point
            self.grabbed_joint = self.joints[joint_name]
            self.grab_point = grab_point

    def release_joint(self, joint_name):
        """Thả một khớp"""
        if joint_name in self.joints:
            self.joints[joint_name].is_grabbed = False
            self.joints[joint_name].grab_point = None
            if self.grabbed_joint == self.joints[joint_name]:
                self.grabbed_joint = None
                self.grab_point = None

    def move_grabbed_joint(self, new_pos):
        """Di chuyển khớp đang được nắm"""
        if self.grabbed_joint is not None and self.grab_point is not None:
            # Tính vector di chuyển
            movement = np.array(new_pos) - np.array(self.grab_point)
            self.grabbed_joint.pos += movement
            self.grab_point = np.array(new_pos) 