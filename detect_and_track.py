from moviepy.editor import VideoFileClip
import numpy as np
import pickle
from functools import reduce
from collections import deque
from scipy.ndimage.measurements import label
from helper.sliding_window import get_layers, search_windows, add_heat, apply_threshold
from helper.draw import draw_labeled_bboxes

def search_windows_args(image, layers, clf, X_scaler):
    return search_windows(image, layers, clf, X_scaler, color_space="HSV", spatial_feat=False, spatial_size=(16,16))

def push_pop(heatmaps, heat):
    heatmaps.append(heat)
    if len(heatmaps) > 4:
        heatmaps.popleft()
    return heatmaps

def avarage_heat(heatmaps):
    h = list(heatmaps)
    if len(h) > 1:
        res = reduce(np.add, h)
    else:
        res = h[0]
    return res, len(h)

def process_image(image):
    global clf, X_scaler, heatmaps
    heat = np.zeros_like(image[:, :, 0]).astype(np.float)
    layers = get_layers(image)
    found = search_windows_args(image, layers, clf, X_scaler)
    heat = add_heat(heat, found)
    heatmaps = push_pop(heatmaps, heat)
    av_h, len = avarage_heat(heatmaps)
    av_h = apply_threshold(av_h, 2+len)
    heatmap = np.clip(av_h, 0, 255)
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(image), labels)
    return draw_img

with open('svc.p', 'rb') as f:
    clf = pickle.load(f)
with open('scaler.p', 'rb') as f:
    X_scaler = pickle.load(f)

heatmaps = deque([])

clip1 = VideoFileClip("./project_video.mp4", audio=False)
print("--> start processing")
white_clip = clip1.fl_image(process_image)
white_clip.write_videofile("./result_video.mp4", audio=False)
print("--> finished")