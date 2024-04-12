from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from urllib import request
from tensorflow.keras import initializers
np.set_printoptions(suppress=True)

# Download test image

def to_VOC_format(width, height, center_x, center_y):
    """
    Convert center coordinate format to min max coordinateformat
    """
    x_min = center_x - 0.5 * width
    y_min = center_y - 0.5 * height
    x_max = center_x + 0.5 * width
    y_max = center_y + 0.5 * height
    return x_min, y_min, x_max, y_max

def to_center_format(xmin_list, ymin_list, xmax_list, ymax_list):
    """
    Convert min max coordinate format to x_center, y_center, height and width format
    """
    height = ymax_list - ymin_list
    width = xmax_list - xmin_list
    
    center_x = xmin_list + 0.5 * width
    center_y = ymin_list + 0.5 * height
    
    return width, height, center_x, center_y

def adjust_deltas(anchor_width, anchor_height, anchor_center_x, anchor_center_y, dx, dy, dw, dh):
    """
    Adjust the anchor box with predicted offset
    """
    # ctr_x = dx * anc_width[:, np.newaxis] + anc_ctr_x[:, np.newaxis]
    center_x = dx * anchor_width + anchor_center_x 
    
    # ctr_y = dy * anc_height[:, np.newaxis] + anc_ctr_y[:, np.newaxis]
    center_y = dy *  anchor_height + anchor_center_y
    
    # w = np.exp(dw) * anc_width[:, np.newaxis]
    width = np.exp(dw) * anchor_width
    
    # np.exp(dh) * anc_height[:, np.newaxis]
    height = np.exp(dh) * anchor_height
    
    return width, height, center_x, center_y

def compute_deltas(base_center_x, base_center_y, base_width, base_height, inside_anchor_width, inside_anchor_height, inside_anchor_center_x, inside_anchor_center_y):
    """
    computing offset of achor box to the groud truth box
    """
    dx = (base_center_x - inside_anchor_center_x)/ inside_anchor_width  # difference in centers of ground truth and anchor box across x axis
    dy = (base_center_y - inside_anchor_center_y)/  inside_anchor_height  # difference in centers of ground truth and anchor box across y axis
    dw = np.log(base_width/ inside_anchor_width) # log on ratio between ground truth width and anchor box width
    dh = np.log(base_height/ inside_anchor_height) # log on ratio between ground truth height and anchor box height
    return dx, dy, dw, dh

# load the image
img = cv2.imread(r'/home/harsh/Downloads/test.jpeg')
# change the format
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# print the shape of the image
print(img.shape)
img_h, img_w, _ = img.shape
plt.imshow(img)
# creating ground truth box and labels for each ground truth
bbox_list = np.array([[120, 25, 200, 165], [300, 50, 480, 320]]) # xmin, ymax, xmax, ymin
plt.show()
# labels 1 for zebra, 0 for background
labels = np.array([1, 1])

# visualize ground truth box
img_ = np.copy(img)
for i, bbox in enumerate(bbox_list):
    cv2.rectangle(img_, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(0, 255, 0), thickness=3) 
plt.imshow(img_)

h = w = 800
# resize input to 800 x 800 (so that we could feed it to pre trained image classification model to extract features)
img = cv2.resize(img, dsize=(h, w), interpolation=cv2.INTER_CUBIC)
# reshape annotations accordingly
x_alter = w/img_w # compression or expansion rate in x axis (expansion in our case, because we went up in x axis, from 333 to 800)
y_alter = h/img_h # compression or expansion rate in y axis (expansion in our case, because we went up in x axis, from 500 to 800)

print(x_alter, y_alter)

# compress or expand ground truth box accordinly
bbox_list[:, 0] = bbox_list[:, 0] * x_alter
bbox_list[:, 1] = bbox_list[:, 1] * y_alter
bbox_list[:, 2] = bbox_list[:, 2] * x_alter
bbox_list[:, 3] = bbox_list[:, 3] * y_alter

img_ = np.copy(img)
for i, bbox in enumerate(bbox_list):
    cv2.rectangle(img_, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(0, 255, 0), thickness=3) 
plt.imshow(img_)

vgg = keras.applications.VGG16(
    include_top=False,
    weights="imagenet"
)

backbone = Model(inputs= [vgg.layers[0].input], outputs= [vgg.layers[17].output])
# examining the shape of feature map
feature_maps = backbone.predict(np.expand_dims(img, 0))
_, w_feature_map, h_feature_map, _ = feature_maps.shape
# number of posssible anchor positions
n_anchor_pos = w_feature_map * h_feature_map
# generating these 2500 centers on the image, placing each of 2500 anchor location with fixed stride to match feature map shape (25*25)

# width stride on the image
x_stride = int(w / w_feature_map)
y_stride = int(h / h_feature_map)

# center (xy coordinate) of anchor location on image
x_center = np.arange(8, w, x_stride) # [  0,  32,  64,  96, 128, 160, 192,...]
y_center = np.arange(8, h, y_stride) # [  0,  32,  64,  96, 128, 160, 192,...]

# generate all the ordered pair of x and y center

# to achive this, we will use meshgrid and reshape it
center_list = np.array(np.meshgrid(x_center, y_center,  sparse=False, indexing='xy')).T.reshape(-1,2)

# visualizing the anchor positions
img_ = np.copy(img)
plt.figure(figsize=(9, 6))
for i in range(n_anchor_pos):
    cv2.circle(img_, (int(center_list[i][0]), int(center_list[i][1])), radius=1, color=(255, 0, 0), thickness=5) 
plt.imshow(img_)

al = []
# aspect ratio = width/ height
anchor_ratio_list = [0.5, 1, 2] # width is half of height(vertical rectangle), width = height (square), width is twice of height (horizontal rectangle)
anchor_scale_list = [8, 16, 32] # area of each anchor box

# total possible anchors 
n_anchors = n_anchor_pos * len(anchor_ratio_list) * len(anchor_scale_list)

# number of object in the image
n_object = len(bbox_list)

# there are total 2500 anchor centers each having 9 anchor boxes placed
# total anchor box in the feature map will be 2500 * 9 = 22500 each anchor box is denoted by 4 numbers.
anchor_list = np.zeros(shape= (n_anchors, 4))

count = 0
# to get height and width given ratio and scale, we will use formula given above
# for each anchor location
for center in center_list:
    center_x, center_y = center[0], center[1]
    # for each ratio
    for ratio in anchor_ratio_list:
        # for each scale
        for scale in anchor_scale_list:
            # compute height and width and scale them by constant factor
            h = pow(pow(scale, 2)/ ratio, 0.5)
            w = h * ratio

            # as h and w would be really small, we will scale them with some constant (in our case, stride width and height)
            h *= x_stride
            w *= y_stride


            # * at this point we have height and width of anchor and centers of anchor locations
            # putting anchor 9 boxes at each anchor locations
            anchor_xmin = center_x - 0.5 * w
            anchor_ymin = center_y - 0.5 * h
            anchor_xmax = center_x + 0.5 * w
            anchor_ymax = center_y + 0.5 * h
            al.append([center_x, center_y, w, h])
            # append the anchor box to anchor list
            anchor_list[count] = [anchor_xmin, anchor_ymin, anchor_xmax, anchor_ymax]
            count += 1

# visualize anchor boxs at center anchor location
img_ = np.copy(img)
# mid anchor center = 2500/2 = 1250
for i in range(11025, 11034):  # 1250 * 9 = 11025 (9 anchors corresponds to mid anchor center)
    x_min = int(anchor_list[i][0])
    y_min = int(anchor_list[i][1])
    x_max = int(anchor_list[i][2])
    y_max = int(anchor_list[i][3])
    cv2.rectangle(img_, (x_min, y_min), (x_max, y_max), color=(255, 0, 0), thickness=3) 

for i, bbox in enumerate(bbox_list):
    cv2.rectangle(img_, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(0, 255, 0), thickness=3)
    
cv2.circle(img_, (int(center_list[312][0]), int(center_list[312][1])), radius=1, color=(0, 0, 255), thickness=15) 

plt.imshow(img_)

h = w = 800
# select anchor boxes which are inside the image
inside_anchor_idx_list = np.where(
    (anchor_list[:,0] >= 0) &
    (anchor_list[:,1] >= 0) &
    (anchor_list[:,2] <= w) &
    (anchor_list[:,3] <= h))[0]
print(inside_anchor_idx_list.shape)
inside_anchor_list = anchor_list[inside_anchor_idx_list]
n_inside_anchor = len(inside_anchor_idx_list)

def IOU(box1, box2):
    """
    Compute overlap (IOU) between box1 and box2
    """
    
    # ------calculate coordinate of overlapping region------
    # take max of x1 and y1 out of both boxes
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    
    # take min of x2 and y2 out of both boxes
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # check if they atleast overlap a little
    if (x1 < x2 and y1 < y2):
        # ------area of overlapping region------
        width_overlap = (x2 - x1)
        height_overlap = (y2 - y1)
        area_overlap = width_overlap * height_overlap
    else:
        # there is no overlap
        return 0
    
    # ------computing union------
    # sum of area of both the boxes - area_overlap
    
    # height and width of both boxes
    width_box1 = (box1[2] - box1[0])
    height_box1 = (box1[3] - box1[1])
    
    width_box2 = (box2[2] - box2[0])
    height_box2 = (box2[3] - box2[1])
    
    # area of box1 and box2
    area_box1 = width_box1 * height_box1
    area_box2 = width_box2 * height_box2
    
    # union (including 2 * overlap area (double count))
    area_union_overlap = area_box1 + area_box2
    
    # union
    area_union = area_union_overlap - area_overlap
    
    # compute IOU
    iou = area_overlap/ area_union
    
    return iou

# computing IOU for each anchor boxes with each ground truth box
iou_list = np.zeros((n_inside_anchor, n_object))
# for each ground truth box
for gt_idx, gt_box in enumerate(bbox_list):
    # for each anchor boxes
    for anchor_idx, anchor_box in enumerate(inside_anchor_list):
        # compute IOU
        iou_list[anchor_idx][gt_idx] = IOU(gt_box, anchor_box)
# convert to dataframe

# add anchor_id
data = {"anchor_id" :inside_anchor_idx_list}

# add object column and corresponding IOU
data.update({f"object_{idx}_iou":iou_list[:, idx] for idx in range(n_object)})

# for each anchor box assign max IOU among all objects in the image
data["max_iou"] = iou_list.max(axis= 1)

# for each anchorbox assign ground truth having maximum IOU
data["best_gt"] = iou_list.argmax(axis= 1)

df_iou = pd.DataFrame(data)
# getting anchor boxes having maximum IOU for each ground truth boxes
best_ious = df_iou.drop(["anchor_id", "max_iou", "best_gt"],axis= 1).max().values
print(f"Top IOUs for each object in the image: {best_ious}")

# getting anchor box idx having maximum overlap with ground truth boxes * ignoring anchor id column
best_anchors = df_iou.drop(["anchor_id", "max_iou", "best_gt"],axis= 1).values.argmax(axis= 0)
print(f"Top anchor boxes index: {best_anchors}")

# get all the anchor boxes having same IOU score
top_anchors = np.where(iou_list == best_ious)[0]
print(f"Anchor boxes with same IOU score: {top_anchors}")
# visualizing top anchor boxes
img_ = np.copy(img)
for i in top_anchors:  # 5625// 2
    x_min = int(inside_anchor_list[i][0])
    y_min = int(inside_anchor_list[i][1])
    x_max = int(inside_anchor_list[i][2])
    y_max = int(inside_anchor_list[i][3])
    cv2.rectangle(img_, (x_min, y_min), (x_max, y_max), color=(255, 0, 0), thickness=3) 

for i, bbox in enumerate(bbox_list):
    cv2.rectangle(img_, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(0, 255, 0), thickness=3)

plt.imshow(img_)

# labeling each anchor box based on IOU threshold

# create dummy column
label_column = np.zeros(df_iou.shape[0], dtype= np.int8)
label_column.fill(-1)

# label top anchor boxes as 1 # contains object
label_column[top_anchors] = 1

# label anchor boxes having IOU > 0.7 with ground truth boxes as 1 # contains object
label_column[np.where(df_iou.max_iou.values >= 0.7)[0]] = 1

# label anchor boxes having IOU < 0.3 with ground truth boxes as 0 # background
label_column[np.where(df_iou.max_iou.values < 0.3)[0]] = 0

# add column to the iou dataframe
df_iou["label"] = label_column


def sample_anchors_pre(df, n_samples= 256, neg_ratio= 0.5):
    """
    Sample total of n samples across both the class (background and foreground),
    If one of the class have less samples than n/2, we will sample from majority class to make up for short.
    """
    n_foreground = int((1-neg_ratio) * n_samples)
    n_backgroud = int(neg_ratio * n_samples)
    foreground_index_list = df[df.label == 1].index.values
    background_index_list = df[df.label == 0].index.values

    # check if we have excessive positive samples
    if len(foreground_index_list) > n_foreground:
        # mark excessive samples as -1 (ignore)
        ignore_index = foreground_index_list[n_backgroud:]
        df.loc[ignore_index, "label"] = -1

    # sample background examples if we don't have enough positive examples to match the anchor batch size
    if len(foreground_index_list) < n_foreground:
        diff = n_foreground - len(foreground_index_list)
        # add remaining to background examples
        n_backgroud += diff

    # check if we have excessive background samples
    if len(background_index_list) > n_backgroud:
        # mark excessive samples as -1 (ignore)
        ignore_index = background_index_list[n_backgroud:]
        df.loc[ignore_index, "label"] = -1

# for each valid anchor box coordinate, convert coordinate format
inside_anchor_width, inside_anchor_height, inside_anchor_center_x, inside_anchor_center_y =  to_center_format(
    inside_anchor_list[:, 0], 
    inside_anchor_list[:, 1],
    inside_anchor_list[:, 2],
    inside_anchor_list[:, 3])


# for each ground truth box corresponds to each anchor box coordinate, convert coordinate format
gt_coordinates = []
for idx in df_iou.best_gt:
    gt_coordinates.append(bbox_list[idx])
gt_coordinates = np.array(gt_coordinates)

base_width, base_height, base_center_x, base_center_y =  to_center_format(
    gt_coordinates[:, 0], 
    gt_coordinates[:, 1],
    gt_coordinates[:, 2],
    gt_coordinates[:, 3])

# the code below prevents from "exp overflow"
eps = np.finfo(inside_anchor_width.dtype).eps
inside_anchor_height = np.maximum(inside_anchor_height, eps)
inside_anchor_width = np.maximum(inside_anchor_width, eps)

# computing offset given by above expression
dx = (base_center_x - inside_anchor_center_x)/ inside_anchor_width  # difference in centers of ground truth and anchor box across x axis
dy = (base_center_y - inside_anchor_center_y)/  inside_anchor_height  # difference in centers of ground truth and anchor box across y axis
dw = np.log(base_width/ inside_anchor_width) # log on ratio between ground truth width and anchor box width
dh = np.log(base_height/ inside_anchor_height) # log on ratio between ground truth height and anchor box height

# adding offsets to the df
df_iou["dx"] = dx
df_iou["dy"] = dy
df_iou["dw"] = dw
df_iou["dh"] = dh

# labels for all possible anchors
label_list = np.empty(n_anchors, dtype = np.float32)
label_list.fill(-1)
label_list[df_iou.anchor_id.values] = df_iou.label.values
label_list = np.expand_dims(label_list, 0)
label_list = np.expand_dims(label_list, -1)

print(df_iou)
# Offset for all possible anchors
offset_list = np.empty(shape= anchor_list.shape, dtype= np.float32)
offset_list.fill(0)
offset_list[df_iou.anchor_id.values] = df_iou[["dx", "dy", "dw", "dh"]].values
offset_list = np.expand_dims(offset_list, 0)
# shape of both the array
print(offset_list.shape, label_list.shape)
offset_list_label_list = np.column_stack((offset_list[0], label_list[0]))[np.newaxis,:]
input_shape = (w_feature_map, h_feature_map, 512) # 50x50X512

from rpn import RPN
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
import pytorch_lightning as pl

feature_maps_tensor = torch.tensor(feature_maps, dtype=torch.float32)
offset_list_tensor = torch.tensor(offset_list, dtype=torch.float32)
label_list_tensor = torch.tensor(label_list, dtype=torch.float32)

rpn_model = RPN(w_feature_map, h_feature_map, 9, n_anchors)
# Wrap tensors into DataLoader
dataset = TensorDataset(feature_maps, (offset_list_label_list, label_list))
train_loader = DataLoader(dataset, batch_size=1)

# Define Lightning Trainer
trainer = pl.Trainer(max_epochs=100)

# Train the model
trainer.fit(rpn_model, train_loader)