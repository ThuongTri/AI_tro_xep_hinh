import math
import copy

def get_aabb(shape):
    # Trả về (min_x, max_x, min_y, max_y, min_z, max_z) của bounding box
    # Đơn giản hóa: chỉ dùng pos và size, không tính xoay (angle) để tối ưu hiệu suất
    s = shape.size / 2
    return (
        shape.pos[0] - s,
        shape.pos[0] + s,
        shape.pos[1] - s,
        shape.pos[1] + s,
        shape.pos[2] - s,
        shape.pos[2] + s
    )

def simulate_drop_preview(shape, falling_shapes, ground_y, min_x, max_x, min_z, max_z, gravity):
    # Tạo bản sao shape để mô phỏng
    test_shape = copy.deepcopy(shape)
    test_shape.falling = True
    test_shape.velocity = 0.0
    test_shape.vel_x = 0.0
    test_shape.vel_z = 0.0
    for _ in range(120):  # mô phỏng tối đa 2 giây
        test_shape.velocity += gravity * 0.5
        test_shape.pos[1] += test_shape.velocity
        # Giới hạn biên
        test_shape.pos[0] = max(min_x, min(max_x, test_shape.pos[0]))
        test_shape.pos[2] = max(min_z, min(max_z, test_shape.pos[2]))
        # Va chạm mặt đất
        bottom = test_shape.pos[1] - test_shape.size/2
        if bottom < ground_y:
            test_shape.pos[1] = ground_y + test_shape.size/2
            test_shape.velocity = 0
            break
        # Va chạm các khối khác (AABB)
        my_aabb = get_aabb(test_shape)
        for other in falling_shapes:
            if not other.falling:
                other_aabb = get_aabb(other)
                overlap_x = my_aabb[1] > other_aabb[0] and my_aabb[0] < other_aabb[1]
                overlap_y = my_aabb[3] > other_aabb[2] and my_aabb[2] < other_aabb[3]
                overlap_z = my_aabb[5] > other_aabb[4] and my_aabb[4] < other_aabb[5]
                if overlap_x and overlap_y and overlap_z:
                    top_y = other_aabb[3]
                    if test_shape.pos[1] - test_shape.size/2 < top_y and test_shape.pos[1] > other.pos[1]:
                        test_shape.pos[1] = top_y + test_shape.size/2
                        test_shape.velocity = 0
                        break
                    # Nếu vẫn overlap x/z, đẩy ra ngoài
                    overlap_amt_x = min(my_aabb[1], other_aabb[1]) - max(my_aabb[0], other_aabb[0])
                    overlap_amt_z = min(my_aabb[5], other_aabb[5]) - max(my_aabb[4], other_aabb[4])
                    if overlap_amt_x > 0 and overlap_amt_x >= overlap_amt_z:
                        if test_shape.pos[0] > other.pos[0]:
                            test_shape.pos[0] = other_aabb[1] + test_shape.size/2
                        else:
                            test_shape.pos[0] = other_aabb[0] - test_shape.size/2
                    elif overlap_amt_z > 0:
                        if test_shape.pos[2] > other.pos[2]:
                            test_shape.pos[2] = other_aabb[5] + test_shape.size/2
                        else:
                            test_shape.pos[2] = other_aabb[4] - test_shape.size/2
    return test_shape.pos, test_shape.angle

def update_physics(shape, ground_y, gravity):
    """Cập nhật vật lý cho một shape"""
    if shape.falling:
        shape.velocity += gravity * 0.5
        shape.pos[1] += shape.velocity
        if shape.pos[1] - shape.size/2 < ground_y:
            shape.pos[1] = ground_y + shape.size/2
            shape.velocity = 0
            shape.falling = False
            shape.state = 'idle' 