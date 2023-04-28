import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns; sns.set()  # for plot styling
from yolov3.configs import *


with open(TRAIN_ANNOT_PATH) as f:
    lines = f.readlines()

w,h = [] , []

for line in lines[0:]:
    ligne_boxes=line.strip().split(' ') 
##strip(): This is a method that removes any whitespace characters (such as spaces, tabs, or newlines)
##from the beginning and end of the line string. This is useful because sometimes text can have extra whitespace that we don't want to 
##include in our processing.
##split(' '): This is a method that splits the line string into a list of substrings, using a space character as the separator.
    for ligne_boxe in ligne_boxes[1:]:
        parts = ligne_boxe.strip().split(',')
        x1 = float(parts[0])
        x2 = float(parts[1])
        y1 = float(parts[2])
        y2 = float(parts[3])
        width = abs(x1-x2)
        height = abs(y1-y2)
        w.append(width)
        h.append(height)

##convert the lists of w and h to numpy arrays, so that mathematical operations can be performed on them more efficiently
w=np.asarray(w)
h=np.asarray(h)
     
x=[w,h] ## x here is a list we need to numpy it 
x=np.asarray(x)

##By transposing x, we can get a 2D array where each row corresponds to a bounding box, 
## and the columns correspond to the width and height values of that box 
# we would not have been able to transpose it.

x=x.transpose()

###############   K- Means   ###############

from sklearn.cluster import KMeans  ##imports the KMeans algorithm from the Scikit-learn library

kmeans3 = KMeans(n_clusters=9, random_state=42, n_init=10)  ##creates an instance of the KMeans algorithm
kmeans3.fit(x)
y_kmeans3 = kmeans3.predict(x)  ##y_kmeans3 is a 1D numpy array that contains the predicted cluster index for each bounding box in the dataset.

##########################################
centers3 = kmeans3.cluster_centers_   ##provides the coordinates of the centers of the clusters

yolo_anchor_average=[]
for ind in range (9):
    yolo_anchor_average.append(np.mean(x[y_kmeans3==ind],axis=0))

yolo_anchor_average=np.array(yolo_anchor_average)

##After this code has been executed, the yolo_anchor_average array will contain the mean values of the data points in each of the 9 clusters.
## These mean values will be used to determine the final anchor boxes for object detection in YOLO.
## The resulting yolo_anchor_average list will have 9 rows (one for each cluster) 
##and 2 columns (one for the mean width and one for the mean height).

plt.scatter(x[:, 0], x[:, 1], c=y_kmeans3, s=2, cmap='inferno')
plt.scatter(yolo_anchor_average[:, 0], yolo_anchor_average[:, 1], c='red', s=50);

# Add axis labels and a title
plt.xlabel("X-coordinates: the widths of the bounding boxes")
plt.ylabel("Y-coordinates: the heights of the bounding boxes")
plt.title("The Average Bounding Box Dimensions for each Cluster")

## this code is adjusting the YOLOv3 anchors to work with a specific image size, which is necessary for accurate object detection

yoloV3anchors = yolo_anchor_average
yoloV3anchors[:, 0] =yolo_anchor_average[:, 0] /TRAIN_INPUT_SIZE *YOLO_INPUT_SIZE  ##!!!!!!!
yoloV3anchors[:, 1] =yolo_anchor_average[:, 1] /TRAIN_INPUT_SIZE *YOLO_INPUT_SIZE   ##!!!!!!!
yoloV3anchors = np.rint(yoloV3anchors) ##This rounding ensures that the anchor values are represented as integers

fig, ax = plt.subplots()
for ind in range(9):
    rectangle= plt.Rectangle((YOLO_INPUT_SIZE/2-yoloV3anchors[ind,0]/2,YOLO_INPUT_SIZE/2-yoloV3anchors[ind,1]/2), yoloV3anchors[ind,0],yoloV3anchors[ind,1] , fc='b',edgecolor='b',fill = None)
    ax.add_patch(rectangle)
    
# Add axis labels and a title
ax.set_aspect(1.0)
ax.set_title('My 9 Anchor Boxes')
ax.set_xlabel('Width')
ax.set_ylabel('Height')
    
## Set the minimum and maximum limits for the x-axis and the y-axis
plt.axis([0,YOLO_INPUT_SIZE,0,YOLO_INPUT_SIZE])
plt.show()
## This line of code sorts the yoloV3anchors array in ascending order
## along the rows axis. This will sort the anchor boxes by their width, from smallest to largest.
yoloV3anchors.sort(axis=0)
print("Your custom anchor boxes are {}".format(yoloV3anchors))