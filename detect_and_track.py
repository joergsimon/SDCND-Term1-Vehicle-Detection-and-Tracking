from moviepy.editor import VideoFileClip
import numpy as np
import pickle
from functools import reduce
from collections import deque
from scipy.ndimage.measurements import label
from helper.sliding_window import get_layers, search_windows, add_heat, apply_threshold
from helper.draw import draw_labeled_bboxes, draw_boxes

def search_windows_args(image, layers, clf, X_scaler):
    return search_windows(image, layers, clf, X_scaler, color_space="HLS", spatial_feat=False, hist_range=(0,256), cell_per_block=1)

def push_pop(heatmaps, heat):
    heatmaps.append(heat)
    if len(heatmaps) > 10:
        heatmaps.popleft()
    return heatmaps

def avarage_heat(heatmaps):
    h = list(heatmaps)
    if len(h) > 1:
        res = reduce(np.add, h)
        res = res/len(h)
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
    av_h, l = avarage_heat(heatmaps)
    av_h = apply_threshold(av_h, 0.9)
    heatmap = np.clip(av_h, 0, 255)
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(image), labels)
    return draw_img

with open('all_clfs.p', 'rb') as f:
    all_clfs = pickle.load(f)
    clf_num = 8
    print('set clf to: ', all_clfs[clf_num][1])
    clf = all_clfs[clf_num][1]
with open('scaler.p', 'rb') as f:
    X_scaler = pickle.load(f)

heatmaps = deque([])

clip1 = VideoFileClip("./project_video.mp4", audio=False).subclip(t_start=40, t_end=43)
#clip1 = VideoFileClip("./project_video.mp4", audio=False)
print("--> start processing")
white_clip = clip1.fl_image(process_image)
white_clip.write_videofile("./result_video-test.mp4", audio=False)
print("--> finished")
