"""
Module Hand Tracker - Xử lý nhận diện và theo dõi bàn tay

Chức năng chính:
1. Theo dõi bàn tay qua webcam sử dụng MediaPipe
2. Nhận diện các cử chỉ tay (PICK, DROP, PALM)
3. Chuyển đổi tọa độ 2D của bàn tay thành tọa độ 3D
4. Xử lý và làm mượt các cử chỉ tay

Các class và hàm chính:
- HandTracker: Class chính xử lý theo dõi bàn tay
- hand_to_3d(): Chuyển đổi tọa độ 2D thành 3D
- estimate_hand_area(): Ước lượng diện tích bàn tay
"""

import cv2
import mediapipe as mp
import threading
import math
import numpy as np
from collections import deque
from scipy.signal import savgol_filter
import time
from constants import (
    MIN_HAND_AREA, MAX_HAND_AREA, DEFAULT_HAND_AREA,
    HAND_DETECTION_CONFIDENCE, HAND_TRACKING_CONFIDENCE
)

class HandTracker(threading.Thread):
    """
    Class theo dõi bàn tay, kế thừa từ Thread để chạy song song
    
    Thuộc tính:
    - x, y, z: Tọa độ 3D của ngón trỏ
    - pinch: Khoảng cách giữa ngón cái và ngón trỏ
    - status: Trạng thái hiện tại (PICK/DROP/PALM/NONE)
    - open_palm: Trạng thái bàn tay mở
    """
    def __init__(self):
        super().__init__()
        # Tọa độ và trạng thái
        self.x, self.y, self.z = 0.5, 0.5, 0.0
        self.pinch = 1.0
        self.running = True
        self.status = "NONE"
        self.open_palm = False
        
        # Buffer cho làm mượt
        self.position_buffer = deque(maxlen=9)  # Buffer cho tọa độ (tăng lên 9 để mượt hơn)
        self.pinch_buffer = deque(maxlen=5)     # Buffer cho pinch
        self.pick_buffer = deque(maxlen=7)      # Buffer cho cử chỉ pick
        self.drop_buffer = deque(maxlen=7)      # Buffer cho cử chỉ drop
        self.palm_buffer = deque(maxlen=7)      # Buffer cho cử chỉ palm
        
        # Các ngưỡng và tham số
        self.PICK_THRESHOLD = 0.07
        self.DROP_THRESHOLD = 0.09
        self.PALM_FACING_THRESHOLD = 0.7
        self.PALM_DISTANCE_THRESHOLD = 0.18
        self.FINGER_DISTANCE_THRESHOLD = 0.07
        
        # Thêm các tham số mới cho điều khiển chính xác
        self.MIN_MOVEMENT_THRESHOLD = 0.01  # Ngưỡng chuyển động tối thiểu
        self.MAX_MOVEMENT_THRESHOLD = 0.3   # Ngưỡng chuyển động tối đa
        self.MOVEMENT_SMOOTHING = 0.3       # Hệ số làm mượt chuyển động
        self.ROTATION_SENSITIVITY = 2.0     # Độ nhạy xoay
        self.ZOOM_SENSITIVITY = 1.5         # Độ nhạy zoom
        
        # Thêm các biến mới cho điều khiển
        self.last_position = None
        self.movement_speed = 0.0
        self.rotation_angle = 0.0
        self.zoom_level = 1.0
        self.is_calibrated = False
        self.calibration_points = []
        
        # Kalman filter cho dự đoán vị trí
        self.kalman = cv2.KalmanFilter(6, 3)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0, 0, 0],
                                                [0, 1, 0, 0, 0, 0],
                                                [0, 0, 1, 0, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 0, 1, 0, 0],
                                               [0, 1, 0, 0, 1, 0],
                                               [0, 0, 1, 0, 0, 1],
                                               [0, 0, 0, 1, 0, 0],
                                               [0, 0, 0, 0, 1, 0],
                                               [0, 0, 0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.array([[1, 0, 0, 0, 0, 0],
                                              [0, 1, 0, 0, 0, 0],
                                              [0, 0, 1, 0, 0, 0],
                                              [0, 0, 0, 1, 0, 0],
                                              [0, 0, 0, 0, 1, 0],
                                              [0, 0, 0, 0, 0, 1]], np.float32) * 0.03
        
        # Lưu trữ landmark cuối cùng
        self.last_landmarks = None
        self.last_valid_position = None
        self.frames_without_hand = 0
        self.max_frames_without_hand = 10

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=HAND_DETECTION_CONFIDENCE,
            min_tracking_confidence=HAND_TRACKING_CONFIDENCE
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.cap = cv2.VideoCapture(0)
        self.hand_area = DEFAULT_HAND_AREA
        self.last_hand_pos = None
        self.last_update_time = time.time()
        self.calibration_frames = []
        self.current_landmarks = None

    def start(self):
        """Bắt đầu quá trình theo dõi tay"""
        print("Starting hand tracking...")
        self.calibrate()

    def calibrate(self):
        """Hiệu chỉnh kích thước bàn tay"""
        print("Calibrating hand size...")
        print("Please show your hand to the camera and keep it steady...")
        
        calibration_frames = []
        start_time = time.time()
        frame_count = 0
        
        while time.time() - start_time < 5:  # Tăng thời gian calibration lên 5 giây
            ret, frame = self.cap.read()
            if not ret:
                continue
                
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Vẽ landmarks
                    self.mp_draw.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    # Tính diện tích bàn tay
                    landmarks = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark])
                    min_x, min_y = np.min(landmarks, axis=0)
                    max_x, max_y = np.max(landmarks, axis=0)
                    area = (max_x - min_x) * (max_y - min_y)
                    calibration_frames.append(area)
                    frame_count += 1
            
            # Hiển thị thông tin
            remaining_time = int(5 - (time.time() - start_time))
            cv2.putText(frame, f"Calibrating... {remaining_time}s", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Frames collected: {frame_count}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("Hand Tracking", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        if calibration_frames:
            # Lọc nhiễu bằng cách loại bỏ các giá trị ngoại lai
            calibration_frames = np.array(calibration_frames)
            mean = np.mean(calibration_frames)
            std = np.std(calibration_frames)
            filtered_frames = calibration_frames[
                (calibration_frames > mean - 2*std) & 
                (calibration_frames < mean + 2*std)
            ]
            
            if len(filtered_frames) > 0:
                self.hand_area = np.mean(filtered_frames)
                self.hand_area = max(MIN_HAND_AREA, min(self.hand_area, MAX_HAND_AREA))
                print(f"Calibration completed! Hand area: {self.hand_area:.4f}")
                print(f"Collected {len(calibration_frames)} frames, {len(filtered_frames)} valid frames")
            else:
                print("Calibration failed! No valid frames collected.")
                self.hand_area = DEFAULT_HAND_AREA
        else:
            print("Calibration failed! No frames collected.")
            self.hand_area = DEFAULT_HAND_AREA
        
        self.is_calibrated = True

    def get_hand_landmarks(self):
        """Lấy landmarks của bàn tay hiện tại"""
        ret, frame = self.cap.read()
        if not ret:
            return None
            
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            self.current_landmarks = results.multi_hand_landmarks[0]
            return self.current_landmarks
        else:
            self.current_landmarks = None
            return None

    def update(self):
        """Cập nhật trạng thái theo dõi tay"""
        if not self.is_calibrated:
            return
            
        ret, frame = self.cap.read()
        if not ret:
            return
            
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Vẽ landmarks
                self.mp_draw.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # Tính diện tích bàn tay
                landmarks = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark])
                min_x, min_y = np.min(landmarks, axis=0)
                max_x, max_y = np.max(landmarks, axis=0)
                area = (max_x - min_x) * (max_y - min_y)
                
                # Lấy vị trí bàn tay
                palm = hand_landmarks.landmark[0]
                current_pos = np.array([palm.x, palm.y, palm.z])
                
                # Cập nhật trạng thái
                if self.last_hand_pos is None:
                    self.last_hand_pos = current_pos
                    self.status = "NONE"
                else:
                    # Tính khoảng cách di chuyển
                    movement = np.linalg.norm(current_pos - self.last_hand_pos)
                    
                    # Kiểm tra cử chỉ nắm/thả
                    thumb_tip = hand_landmarks.landmark[4]
                    index_tip = hand_landmarks.landmark[8]
                    distance = np.sqrt((thumb_tip.x - index_tip.x)**2 + 
                                     (thumb_tip.y - index_tip.y)**2)
                    
                    if distance < 0.1:  # Nếu ngón cái và ngón trỏ gần nhau
                        self.status = "PICK"
                    else:
                        self.status = "DROP"
                    
                    self.last_hand_pos = current_pos
                
                # Hiển thị thông tin
                cv2.putText(frame, f"Status: {self.status}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Area: {area:.4f}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("Hand Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False
        
        return True

    def stop(self):
        """Dừng theo dõi tay"""
        self.cap.release()
        cv2.destroyAllWindows()
        self.hands.close()

    def calculate_movement(self, current_pos):
        """Tính toán chuyển động với độ chính xác cao hơn"""
        if self.last_position is None:
            self.last_position = current_pos
            return 0.0, 0.0, 0.0
            
        # Tính vector chuyển động
        movement = np.array(current_pos) - np.array(self.last_position)
        
        # Áp dụng ngưỡng chuyển động
        movement = np.clip(movement, -self.MAX_MOVEMENT_THRESHOLD, self.MAX_MOVEMENT_THRESHOLD)
        
        # Làm mượt chuyển động
        smoothed_movement = movement * self.MOVEMENT_SMOOTHING
        
        # Cập nhật vị trí cuối
        self.last_position = current_pos
        
        return smoothed_movement

    def calculate_rotation(self, hand_landmarks):
        """Tính toán góc xoay từ cử chỉ tay"""
        if not hand_landmarks:
            return 0.0
            
        # Lấy các điểm quan trọng
        wrist = hand_landmarks.landmark[0]
        index_finger = hand_landmarks.landmark[8]
        pinky = hand_landmarks.landmark[20]
        
        # Tính vector từ cổ tay đến ngón trỏ và ngón út
        v1 = [index_finger.x - wrist.x, index_finger.y - wrist.y]
        v2 = [pinky.x - wrist.x, pinky.y - wrist.y]
        
        # Tính góc giữa hai vector
        angle = math.atan2(v2[1], v2[0]) - math.atan2(v1[1], v1[0])
        angle = math.degrees(angle)
        
        # Chuẩn hóa góc về [-180, 180]
        angle = (angle + 180) % 360 - 180
        
        return angle * self.ROTATION_SENSITIVITY

    def calculate_zoom(self, hand_landmarks):
        """Tính toán mức zoom từ khoảng cách giữa các ngón tay"""
        if not hand_landmarks:
            return 1.0
            
        # Lấy các điểm quan trọng
        thumb = hand_landmarks.landmark[4]
        index = hand_landmarks.landmark[8]
        middle = hand_landmarks.landmark[12]
        ring = hand_landmarks.landmark[16]
        pinky = hand_landmarks.landmark[20]
        
        # Tính khoảng cách trung bình giữa các ngón tay
        distances = []
        fingers = [thumb, index, middle, ring, pinky]
        for i in range(len(fingers)-1):
            dist = math.sqrt((fingers[i].x - fingers[i+1].x)**2 + 
                           (fingers[i].y - fingers[i+1].y)**2)
            distances.append(dist)
            
        avg_distance = np.mean(distances)
        
        # Chuẩn hóa khoảng cách về mức zoom
        zoom = 1.0 + (avg_distance - 0.1) * self.ZOOM_SENSITIVITY
        zoom = np.clip(zoom, 0.5, 2.0)
        
        return zoom

    def smooth_position(self, x, y, z):
        """Làm mượt tọa độ sử dụng Savitzky-Golay filter"""
        self.position_buffer.append([x, y, z])
        if len(self.position_buffer) >= 5:
            positions = np.array(self.position_buffer)
            smoothed = savgol_filter(positions, 5, 2, axis=0)
            return smoothed[-1]
        return np.array([x, y, z])

    def predict_position(self):
        """Dự đoán vị trí khi không phát hiện được bàn tay"""
        if self.last_valid_position is not None:
            prediction = self.kalman.predict()
            return prediction[:3].flatten()
        return None

    def update_kalman(self, measurement):
        """Cập nhật Kalman filter với vị trí mới"""
        try:
            # Đảm bảo measurement là float32
            measurement = np.array(measurement, dtype=np.float32).reshape(3, 1)
            self.kalman.correct(measurement)
            self.last_valid_position = measurement.flatten()
        except Exception as e:
            print(f"Lỗi Kalman update: {e}")
            # Nếu có lỗi, sử dụng vị trí trực tiếp
            self.last_valid_position = measurement.flatten()

    def detect_gesture(self, hand_landmarks):
        """Nhận diện cử chỉ tay với độ chính xác cao hơn"""
        if not hand_landmarks or len(hand_landmarks.landmark) != 21:
            return False, False, False

        try:
            # Calibration nếu cần
            if not self.is_calibrated:
                self.calibrate()

            # Lấy các điểm quan trọng
            index_finger = hand_landmarks.landmark[8]
            thumb_finger = hand_landmarks.landmark[4]
            middle_finger = hand_landmarks.landmark[12]
            wrist = hand_landmarks.landmark[0]

            # Tính toán các khoảng cách
            pinch_dist = math.sqrt((index_finger.x - thumb_finger.x) ** 2 + 
                                (index_finger.y - thumb_finger.y) ** 2 + 
                                (index_finger.z - thumb_finger.z) ** 2)
            
            # Cập nhật buffer pinch
            self.pinch_buffer.append(pinch_dist)
            if len(self.pinch_buffer) >= 3:
                self.pinch = np.mean(self.pinch_buffer)

            # Nhận diện cử chỉ PICK
            pick_now = pinch_dist < self.PICK_THRESHOLD
            
            # Nhận diện cử chỉ DROP
            drop_now = pinch_dist > self.DROP_THRESHOLD

            # Nhận diện cử chỉ PALM
            # Tính vector pháp tuyến của bàn tay
            base_index = hand_landmarks.landmark[5]
            base_pinky = hand_landmarks.landmark[17]
            v1 = [base_index.x - wrist.x, base_index.y - wrist.y, base_index.z - wrist.z]
            v2 = [base_pinky.x - wrist.x, base_pinky.y - wrist.y, base_pinky.z - wrist.z]
            normal = [
                v1[1]*v2[2] - v1[2]*v2[1],
                v1[2]*v2[0] - v1[0]*v2[2],
                v1[0]*v2[1] - v1[1]*v2[0]
            ]
            normal_length = math.sqrt(sum(x*x for x in normal))
            if normal_length > 0:
                normal = [x/normal_length for x in normal]
                palm_facing = abs(normal[1]) > self.PALM_FACING_THRESHOLD
            else:
                palm_facing = False

            # Cập nhật các buffer
            self.pick_buffer.append(pick_now)
            self.drop_buffer.append(drop_now)
            self.palm_buffer.append(palm_facing)

            # Xác định trạng thái cuối cùng
            pick = sum(self.pick_buffer) >= len(self.pick_buffer) * 0.7
            drop = sum(self.drop_buffer) >= len(self.drop_buffer) * 0.7
            palm = sum(self.palm_buffer) >= len(self.palm_buffer) * 0.7

            return pick, drop, palm

        except Exception as e:
            print(f"Lỗi detect_gesture: {e}")
            return False, False, False

    def run(self):
        """Hàm chính chạy trong thread"""
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        mp_draw = mp.solutions.drawing_utils

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue

            # Lật frame để hiển thị như gương
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                self.frames_without_hand = 0
                hand_landmarks = results.multi_hand_landmarks[0]
                self.last_landmarks = hand_landmarks

                # Lấy tọa độ ngón trỏ
                index_finger = hand_landmarks.landmark[8]
                x, y, z = index_finger.x, index_finger.y, index_finger.z

                # Chuyển đổi tọa độ 2D thành 3D
                x3d, y3d, z3d = hand_to_3d(x, y, z)
                
                # Làm mượt tọa độ
                smoothed_pos = self.smooth_position(x3d, y3d, z3d)
                
                # Cập nhật Kalman filter
                self.update_kalman(smoothed_pos)
                
                # Cập nhật tọa độ
                self.x, self.y, self.z = smoothed_pos

                # Nhận diện cử chỉ
                pick, drop, palm = self.detect_gesture(hand_landmarks)
                
                # Cập nhật trạng thái
                if pick:
                    self.status = "PICK"
                elif drop:
                    self.status = "DROP"
                elif palm:
                    self.status = "PALM"
                else:
                    self.status = "NONE"

                # Vẽ debug info
                self.draw_debug_info(frame)

            else:
                self.frames_without_hand += 1
                if self.frames_without_hand > self.max_frames_without_hand:
                    # Dự đoán vị trí khi không phát hiện được bàn tay
                    predicted_pos = self.predict_position()
                    if predicted_pos is not None:
                        self.x, self.y, self.z = predicted_pos
                    self.status = "NONE"

            # Hiển thị frame
            cv2.imshow('Hand Tracking', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def draw_debug_info(self, frame):
        """Vẽ thông tin debug lên frame"""
        # Vẽ tọa độ
        cv2.putText(frame, f"X: {self.x:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Y: {self.y:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Z: {self.z:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Vẽ trạng thái
        cv2.putText(frame, f"Status: {self.status}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Vẽ pinch distance
        cv2.putText(frame, f"Pinch: {self.pinch:.2f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

def hand_to_3d(x, y, z, scale=12.0):
    """Chuyển đổi tọa độ 2D của bàn tay thành tọa độ 3D"""
    # Chuẩn hóa tọa độ về [-1, 1]
    x = (x - 0.5) * 2
    y = (y - 0.5) * 2
    z = (z - 0.5) * 2
    
    # Áp dụng scale
    x *= scale
    y *= scale
    z *= scale
    
    return x, y, z

def estimate_hand_area(hand_landmarks):
    """Ước lượng diện tích bàn tay"""
    if not hand_landmarks:
        return 0.0
        
    # Lấy các điểm quan trọng
    wrist = hand_landmarks.landmark[0]
    thumb = hand_landmarks.landmark[4]
    index = hand_landmarks.landmark[8]
    pinky = hand_landmarks.landmark[20]
    
    # Tính diện tích tam giác
    def triangle_area(p1, p2, p3):
        a = math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
        b = math.sqrt((p2.x - p3.x)**2 + (p2.y - p3.y)**2)
        c = math.sqrt((p3.x - p1.x)**2 + (p3.y - p1.y)**2)
        s = (a + b + c) / 2
        return math.sqrt(s * (s - a) * (s - b) * (s - c))
    
    # Tính tổng diện tích
    area = triangle_area(wrist, thumb, index)
    area += triangle_area(wrist, index, pinky)
    
    return area 