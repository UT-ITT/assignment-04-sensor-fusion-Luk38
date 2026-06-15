from pathlib import Path
import argparse
import cv2
import numpy as np

WINDOW_NAME = 'Preview Window'
WARPED_WINDOW_NAME = 'Warped Result'

# parser for inline parameters input and output paths and resolution for the warped image
parser = argparse.ArgumentParser(description='input file and output destination')
parser.add_argument('--input_path', type=str, required=True)
parser.add_argument('--output_path', type=str, required=True)
parser.add_argument('--resolution', type=int, nargs=2, default=[800, 600])
args = parser.parse_args()

image_path = Path(args.input_path)
output_path = Path(args.output_path)
resolution = tuple(args.resolution)

img = cv2.imread(str(image_path))

if img is None:
    raise FileNotFoundError(f'Could not load image: {image_path}')

cv2.namedWindow(WINDOW_NAME)

# copy of the image to not change the original when drawing points and warping
original_img = img.copy()
display_img = img.copy()
selected_points = []
warped_img = None

# order the points for the rectangle
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

# transform the selected rectangle to a new image
def warp_selected_rectangle(points):
    global warped_img

    ordered_points = get_points_in_order(points)
    width, height = resolution
    destination_points = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype='float32',
    )
    transform = cv2.getPerspectiveTransform(ordered_points, destination_points)
    warped_img = cv2.warpPerspective(original_img, transform, (width, height))
    cv2.imshow(WARPED_WINDOW_NAME, warped_img) #show the new image in a new window

# reset function to discard changes and start over with a new selection of points
def reset_selection():
    global display_img, warped_img

    selected_points.clear()
    display_img = original_img.copy()
    warped_img = None
    cv2.imshow(WINDOW_NAME, display_img)
    cv2.destroyWindow(WARPED_WINDOW_NAME)

# get the mouse clicks to select the points
def mouse_callback(event, x, y, flags, param):
    global display_img

    if event != cv2.EVENT_LBUTTONDOWN or len(selected_points) >= 4:
        return

    selected_points.append((x, y))
    display_img = cv2.circle(display_img, (x, y), 5, (255, 0, 0), -1)
    cv2.imshow(WINDOW_NAME, display_img)

    # start warping when 4 points are selected
    if len(selected_points) == 4:
        warp_selected_rectangle(selected_points)

cv2.imshow(WINDOW_NAME, display_img)

cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

# handle key pesses
# ESC to reset, S to save, Q to quit
while True:
    key = cv2.waitKey(20) & 0xFF 

    if key == 27: # ESC
        reset_selection()
        continue

    if key in (ord('s'), ord('S')) and warped_img is not None:
        cv2.imwrite(str(output_path), warped_img)
        break

    if key in (ord('q'), ord('Q')):
        break