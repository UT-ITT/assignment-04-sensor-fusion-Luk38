import cv2
import cv2.aruco as aruco
import numpy as np
import pyglet 
from PIL import Image
from dataclasses import dataclass
import mediapipe as mp 
import random
import time
import sys
from pathlib import Path
from pyglet.window import key 

# MediaPipe hand landmarker setup
MODEL_PATH = Path(__file__).with_name('hand_landmarker.task')
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Initialize MediaPipe Landmarker
landmarker = HandLandmarker.create_from_options(
    HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
)

video_id = 0

if len(sys.argv) > 1:
    video_id = int(sys.argv[1])

# Define the ArUco dictionary, parameters, and detector
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
aruco_params = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, aruco_params)

# converts OpenCV image to PIL image and then to pyglet texture
# https://gist.github.com/nkymut/1cb40ea6ae4de0cf9ded7332f1ca0d55
def cv2glet(img,fmt):
    '''Assumes image is in BGR color space. Returns a pyimg object'''
    if fmt == 'GRAY':
      rows, cols = img.shape
      channels = 1
    else:
      rows, cols, channels = img.shape

    raw_img = Image.fromarray(img).tobytes()

    top_to_bottom_flag = -1
    bytes_per_row = channels*cols
    pyimg = pyglet.image.ImageData(width=cols, 
                                   height=rows, 
                                   fmt=fmt, 
                                   data=raw_img, 
                                   pitch=top_to_bottom_flag*bytes_per_row)
    return pyimg

# Create a video capture object for the webcam
cap = cv2.VideoCapture(video_id)

if not cap.isOpened():
    raise RuntimeError('Could not open the webcam')

ret, initial_frame = cap.read()
if not ret:
    raise RuntimeError('Could not read a frame from the webcam')

WINDOW_HEIGHT, WINDOW_WIDTH = initial_frame.shape[:2]

window = pyglet.window.Window(WINDOW_WIDTH, WINDOW_HEIGHT)

FINGER_SMOOTHING = 0.8
FINGER_PADDING_X = 42
FINGER_PADDING_Y = 12
PADDLE_WIDTH = 110
PADDLE_HEIGHT = 16
BALL_RADIUS = 12
BALL_BASE_SPEED = 200.0
MIRROR_BOARD = True


@dataclass
class Ball:
    x: float
    y: float
    vx: float
    vy: float


game_start_time = time.monotonic()
latest_display_frame = initial_frame.copy()
latest_finger_pos = None
smoothed_finger_pos = None
ball = None
score = 0
# Helper functions for clamping values and ordering points
def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))

def get_points_in_order(points):
    rectangle = np.zeros((4, 2), dtype='float32')
    points = np.array(points, dtype='float32')

    point_sums = points.sum(axis=1)
    point_diffs = np.diff(points, axis=1)

    rectangle[0] = points[np.argmin(point_sums)]
    rectangle[2] = points[np.argmax(point_sums)]
    rectangle[1] = points[np.argmin(point_diffs)]
    rectangle[3] = points[np.argmax(point_diffs)]

    return rectangle

def get_board_corners(marker_corners):
    if len(marker_corners) < 4:
        return None

    marker_centers = np.array([
        np.mean(marker[0], axis=0) for marker in marker_corners
    ], dtype='float32')

    point_sums = marker_centers.sum(axis=1)
    point_diffs = np.diff(marker_centers, axis=1)

    ordered_marker_indices = [
        int(np.argmin(point_sums)),
        int(np.argmin(point_diffs)),
        int(np.argmax(point_sums)),
        int(np.argmax(point_diffs)),
    ]

    # OpenCV returns each marker corner as top-left, top-right, bottom-right, bottom-left
    inner_corner_indices = [2, 3, 0, 1]
    board_corners = []

    for marker_index, corner_index in zip(ordered_marker_indices, inner_corner_indices):
        board_corners.append(marker_corners[marker_index][0][corner_index])

    return np.array(board_corners, dtype='float32')


def warp_board(frame, marker_corners):
    board_corners = get_board_corners(marker_corners)

    if board_corners is None:
        return None, None

    ordered_board_corners = get_points_in_order(board_corners)

    destination_points = np.array([
        [0, 0],
        [WINDOW_WIDTH - 1, 0],
        [WINDOW_WIDTH - 1, WINDOW_HEIGHT - 1],
        [0, WINDOW_HEIGHT - 1],
    ], dtype='float32')

    transform = cv2.getPerspectiveTransform(
        ordered_board_corners,
        destination_points
    )

    warped = cv2.warpPerspective(
        frame,
        transform,
        (WINDOW_WIDTH, WINDOW_HEIGHT)
    )

    return warped, transform

def create_ball():
    return Ball(
        x=WINDOW_WIDTH / 2.0,
        y=BALL_RADIUS + 20,
        vx=random.choice([-1.0, 1.0]) * BALL_BASE_SPEED,
        vy=BALL_BASE_SPEED,
    )

def reset_round():
    global ball, score

    ball = create_ball()
    score = 0

# fingertip detection function with mediapipe
def detect_fingertip(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    timestamp_ms = int((time.monotonic() - game_start_time) * 1000)
    results = landmarker.detect_for_video(mp_image, timestamp_ms)

    if not results.hand_landmarks:
        return None

    fingertip = results.hand_landmarks[0][8]
    height, width = frame.shape[:2]
    return int(np.clip(fingertip.x * width, 0, width - 1)), int(np.clip(fingertip.y * height, 0, height - 1))

# Ball movement, collision, and scoring function
def update_ball(active_ball, dt, paddle_rect):
    global score
    dt = min(dt, 0.05)
    substeps = 4
    step_dt = dt / substeps
    for _ in range(substeps):
        active_ball.x += active_ball.vx * step_dt
        active_ball.y += active_ball.vy * step_dt

        # Check for collisions with the walls
        if active_ball.x - BALL_RADIUS <= 0:
            active_ball.x = BALL_RADIUS
            active_ball.vx = abs(active_ball.vx)
        elif active_ball.x + BALL_RADIUS >= WINDOW_WIDTH:
            active_ball.x = WINDOW_WIDTH - BALL_RADIUS
            active_ball.vx = -abs(active_ball.vx)

        if active_ball.y - BALL_RADIUS <= 0:
            active_ball.y = BALL_RADIUS
            active_ball.vy = abs(active_ball.vy)
            score += 1
            return

        paddle_x, paddle_y = paddle_rect
        paddle_left = paddle_x - PADDLE_WIDTH / 2.0
        paddle_right = paddle_x + PADDLE_WIDTH / 2.0
        paddle_top = paddle_y + PADDLE_HEIGHT / 2.0
        paddle_bottom = paddle_y - PADDLE_HEIGHT / 2.0

        # Check for collision with the paddle
        if (
            active_ball.x + BALL_RADIUS >= paddle_left and
            active_ball.x - BALL_RADIUS <= paddle_right and
            active_ball.y + BALL_RADIUS >= paddle_bottom and
            active_ball.y - BALL_RADIUS <= paddle_top and
            active_ball.vy > 0
        ):

            active_ball.y = paddle_top - BALL_RADIUS - 1

            active_ball.vy = -abs(active_ball.vy)

            offset = (active_ball.x - paddle_x) / (PADDLE_WIDTH / 2.0)

            active_ball.vx += offset * 140.0

            active_ball.vx = clamp(
                active_ball.vx,
                -BALL_BASE_SPEED * 1.6,
                BALL_BASE_SPEED * 1.6
            )

            return

        if active_ball.y + BALL_RADIUS >= WINDOW_HEIGHT:
            reset_round()
            return


def draw_paddle(frame, paddle_pos):
    paddle_x, paddle_y = paddle_pos
    top_left = (
        int(clamp(paddle_x - PADDLE_WIDTH / 2.0, 0, WINDOW_WIDTH - 1)),
        int(clamp(paddle_y - PADDLE_HEIGHT / 2.0, 0, WINDOW_HEIGHT - 1)),
    )
    bottom_right = (
        int(clamp(paddle_x + PADDLE_WIDTH / 2.0, 0, WINDOW_WIDTH - 1)),
        int(clamp(paddle_y + PADDLE_HEIGHT / 2.0, 0, WINDOW_HEIGHT - 1)),
    )
    cv2.rectangle(frame, top_left, bottom_right, (0, 0, 0), -1)
    cv2.rectangle(frame, top_left, bottom_right, (255, 255, 255), 2)


def draw_ball(frame, active_ball):
    cv2.circle(frame, (int(active_ball.x), int(active_ball.y)), BALL_RADIUS, (0, 255, 255), -1)
    cv2.circle(frame, (int(active_ball.x), int(active_ball.y)), BALL_RADIUS, (255, 255, 255), 2)

# user interface
def draw_ui(frame, elapsed_seconds):
    cv2.putText(frame, f'Score: {score}', (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)


reset_hint = 'Show all four corner markers to start'


def process_frame(frame, dt):
    global latest_finger_pos, latest_display_frame, smoothed_finger_pos, ball

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    
    # warp the view between the markers
    warped_frame, transform = warp_board(frame, corners) if ids is not None and len(corners) > 0 else (None, None)

    if warped_frame is not None:
        mirrored_frame = cv2.flip(warped_frame, -1) if MIRROR_BOARD else warped_frame
        detected_fingertip = detect_fingertip(frame)

        # correct finger position based on the perspective and mirroring of the board
        if detected_fingertip is not None:
            pt = np.array([[[detected_fingertip[0], detected_fingertip[1]]]], dtype=np.float32)

            board_pt = cv2.perspectiveTransform(pt, transform)[0][0]
            board_pt[0] = WINDOW_WIDTH - board_pt[0]
            board_pt[1] = WINDOW_HEIGHT - board_pt[1]

            detected_fingertip = board_pt
            if smoothed_finger_pos is None:
                smoothed_finger_pos = detected_fingertip
            else:
                smoothed_finger_pos = (
                    FINGER_SMOOTHING * smoothed_finger_pos +
                    (1.0 - FINGER_SMOOTHING) * detected_fingertip
                )
            latest_finger_pos = (int(smoothed_finger_pos[0]), int(smoothed_finger_pos[1]))
        else:
            latest_finger_pos = None
        if ball is None:
            ball = create_ball() # initialize the ball 

        # position of the paddle based on finger position
        if latest_finger_pos is not None:
            paddle_reference = latest_finger_pos
        elif smoothed_finger_pos is not None:
            paddle_reference = (int(smoothed_finger_pos[0]), int(smoothed_finger_pos[1]))
        else:
            paddle_reference = (WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2)

        paddle_x = clamp(
            paddle_reference[0],
            PADDLE_WIDTH / 2.0 + FINGER_PADDING_X,
            WINDOW_WIDTH - PADDLE_WIDTH / 2.0 - FINGER_PADDING_X,
        )
        paddle_y = clamp(
            paddle_reference[1],
            PADDLE_HEIGHT / 2.0 + FINGER_PADDING_Y,
            WINDOW_HEIGHT - PADDLE_HEIGHT / 2.0 - FINGER_PADDING_Y,
        )

        update_ball(ball, dt, (paddle_x, paddle_y))

    # draw the game elements on top of the warped frame
        board_frame = mirrored_frame.copy()
        draw_ball(board_frame, ball)
        draw_paddle(board_frame, (paddle_x, paddle_y))
        if latest_finger_pos is not None:
            cv2.circle(board_frame, latest_finger_pos, 6, (255, 255, 255), -1)
    else:
        latest_finger_pos = None
        board_frame = frame.copy()

    if warped_frame is None:
        cv2.putText(board_frame, reset_hint, (15, WINDOW_HEIGHT - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    draw_ui(board_frame, time.monotonic() - game_start_time)

    latest_display_frame = board_frame

# process and update camera frame
def update(dt):
    ret, frame = cap.read()
    if not ret:
        return

    process_frame(frame, dt)

@window.event
def on_key_press(symbol, modifiers):
    if symbol in (key.Q, key.ESCAPE): # quit on Q or ESC
        cap.release()
        landmarker.close()
        window.close()
        pyglet.app.exit()

pyglet.clock.schedule_interval(update, 1 / 30)

@window.event
def on_draw():
    window.clear()

    img = cv2glet(latest_display_frame, 'BGR')
    img.blit(0, 0, 0)

reset_round()
pyglet.app.run()
cap.release()
landmarker.close()