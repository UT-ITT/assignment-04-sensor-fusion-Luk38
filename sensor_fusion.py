import cv2
import cv2.aruco as aruco
import numpy as np
import pyglet
from pyglet.window import key
from PIL import Image
import sys

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
latest_display_frame = initial_frame.copy()

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

def process_frame(frame, dt):
    global latest_display_frame

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    
    # warp the view between the markers
    warped_frame, transform = warp_board(frame, corners) if ids is not None and len(corners) > 0 else (None, None)

    if warped_frame is not None:        
        board_frame = warped_frame.copy()
    else:
        latest_finger_pos = None
        board_frame = frame.copy()

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
        window.close()
        pyglet.app.exit()

pyglet.clock.schedule_interval(update, 1 / 30)

@window.event
def on_draw():
    window.clear()

    img = cv2glet(latest_display_frame, 'BGR')
    img.blit(0, 0, 0)


pyglet.app.run()
cap.release()