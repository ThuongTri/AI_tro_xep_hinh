# Display settings
DISPLAY_SIZE = (800, 600)
FPS = 60

# Game settings
GRAVITY = -9.8
GROUND_Y = -2.0
PICK_COOLDOWN_FRAMES = 10
DROP_COOLDOWN_FRAMES = 10
MAX_ANGLE_STEP = 8.0
SCALE_MAPPING = 15.0

# Hand tracking settings
HAND_DETECTION_CONFIDENCE = 0.7
HAND_TRACKING_CONFIDENCE = 0.5
MIN_HAND_AREA = 0.01
MAX_HAND_AREA = 0.5
DEFAULT_HAND_AREA = 0.1

# Mannequin settings
MANNEQUIN_SCALE = 1.0
JOINT_RADIUS = 0.1
LINK_RADIUS = 0.05

# Colors
WHITE = (1.0, 1.0, 1.0)
RED = (1.0, 0.0, 0.0)
GREEN = (0.0, 1.0, 0.0)
BLUE = (0.0, 0.0, 1.0)
GRAY = (0.5, 0.5, 0.5)

# Camera settings
CAMERA_DISTANCE = 10.0
CAMERA_FOV = 45.0
CAMERA_NEAR = 0.1
CAMERA_FAR = 50.0

# Shape definitions
SHAPES = ['cube', 'sphere', 'pyramid', 'rectangle']
SHAPE_COLORS = {
    'cube': [(0.9, 0.5, 0.5), (0.5, 0.9, 0.5), (0.5, 0.7, 0.9), (0.95, 0.95, 0.6), (0.8, 0.5, 0.9), (0.5, 0.9, 0.9)],
    'sphere': [(0.8, 0.3, 0.3), (0.3, 0.8, 0.3), (0.3, 0.3, 0.8)],
    'pyramid': [(0.9, 0.4, 0.4), (0.4, 0.9, 0.4), (0.4, 0.4, 0.9), (0.9, 0.9, 0.4)],
    'rectangle': [(0.7, 0.5, 0.5), (0.5, 0.7, 0.5), (0.5, 0.5, 0.7), (0.7, 0.7, 0.5)]
}

# Các đỉnh của các hình khối
CUBE_VERTICES = [
    [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
    [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1]
]

PYRAMID_VERTICES = [
    [0, 1, 0],    # top
    [-1, -1, -1], # front left
    [1, -1, -1],  # front right
    [1, -1, 1],   # back right
    [-1, -1, 1]   # back left
]

RECTANGLE_VERTICES = [
    [1.5, 1, -1], [1.5, -1, -1], [-1.5, -1, -1], [-1.5, 1, -1],
    [1.5, 1, 1], [1.5, -1, 1], [-1.5, -1, 1], [-1.5, 1, 1]
]

# Các cạnh của các hình khối
CUBE_EDGES = (
    (0,1),(1,2),(2,3),(3,0),
    (4,5),(5,6),(6,7),(7,4),
    (0,4),(1,5),(2,6),(3,7)
)

PYRAMID_EDGES = (
    (0,1),(0,2),(0,3),(0,4),
    (1,2),(2,3),(3,4),(4,1)
)

RECTANGLE_EDGES = (
    (0,1),(1,2),(2,3),(3,0),
    (4,5),(5,6),(6,7),(7,4),
    (0,4),(1,5),(2,6),(3,7)
)

# Các mặt của các hình khối
CUBE_SURFACES = (
    (0,1,2,3), (4,5,6,7), (0,1,5,4),
    (2,3,7,6), (1,2,6,5), (0,3,7,4)
)

PYRAMID_FACES = (
    (0,1,2), (0,2,3), (0,3,4), (0,4,1),
    (1,2,3,4)
)

RECTANGLE_SURFACES = (
    (0,1,2,3), (4,5,6,7), (0,1,5,4),
    (2,3,7,6), (1,2,6,5), (0,3,7,4)
) 