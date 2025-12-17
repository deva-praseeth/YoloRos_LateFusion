# Late fusion of YOLOv8 Detections
This repository contains the code base for the fusion of object detections using YOLOv8 detections from 5 different Lucid Vision GigE cameras. We run docker container with tmuxinator panes to perform YOLO detections on each camera individually and the fusion node, running in the host, performs late fusion of the detection topics. Moreover, fusion node performs a 2x3 stitch of the 5 camera detection images (possible 6th camera addition).

## Table of Contents

1. [Installation](#installation)
2. [Docker](#docker)
3. [Late Fusion](#late-fusion)
4. [ROS2 Wrap for YOLO](#ros2-wrap-for-yolo)
5. [Models](#models)
6. [Usage](#usage)
7. [Demos](#demos)

## Installation

```shell
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
git clone https://github.com/deva-praseeth/YoloRos_LateFusion.git
# if Ubuntu ver.>=24.04, perform below commands in venv (recommended), or use --break-system-packages
pip3 install -r YoloRos_LateFusion/requirements.txt
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build
```

## Docker

Build the yolo_ros docker.

```shell
cd ~/ros2_ws/src/YoloRos_LateFusion
docker build -t yolo_ros .
```

If you want to use CUDA, you have to install the [NVIDIA Container Tollkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) and add `--gpus all`.

Each container will subscribe to individual cameras and publish detection image, hence if there are 5 cameras, 5 containers should be created with the following with different names.

```shell
sudo docker run -it \
  --gpus all \
  --cap-add=ALL \
  --device=/dev/bus/usb:/dev/bus/usb \
  --network=host \
  --hostname <host-name> \ #host username
  -v /home:/host_home \
  -v /etc:/host_etc \
  -v /var:/host_var \
  -w /root \
  --name <name> \ #<name> = yolo_cam1, yolo_cam2...
  yolo_ros

```

Verify that container is created
```shell
sudo docker ps -a
```
Start the container
```shell
sudo docker start -ai <container name>
colcon build --packages-select yolo_ros
source install/setup.sh
ros2 launch yolo_bringup yolov8.launch.py
```

## Late Fusion
While the yolo_detections are running in 5 containers, the fusion_node.py code can be run on the native host. The node subscribes to all the /yolo/detections and append detection messages from all the camera to send /fused/detections at 10hz. 

Moreover, the node subscribes to all /yolo/dbg_image and stitches the images in 2x3 panorama for rviz visualization. It publishes a black tile in place of cameras with inactive feed.
Below given is an example of /fused/detections messages.
```shell
ros2 topic echo /fused/detections --once
  header:
    stamp:
      sec: 1760620602
      nanosec: 586551826
    frame_id: fused
  detections:
  - class_id: 9
    class_name: traffic light
    score: 0.8731038570404053
    id: ''
    bbox:
      center:
        position:
          x: 368.0845642089844
          y: 131.0710906982422
        theta: 0.0
      size:
        x: 129.17218017578125
        y: 260.4407653808594
    bbox3d:
      center:
        position:
          x: 0.0
          y: 0.0
          z: 0.0
        orientation:
          x: 0.0
          y: 0.0
          z: 0.0
          w: 1.0
      size:
        x: 0.0
        y: 0.0
        z: 0.0
      frame_id: camera_2/camera_link
    mask:
      height: 0
      width: 0
      data: []
    keypoints:
      data: []
    keypoints3d:
      data: []
      frame_id: camera_2/camera_link
  - class_id: 9
    class_name: traffic light
    score: 0.7921506762504578
    id: ''
    bbox:
      center:
        position:
          x: 485.69000244140625
          y: 128.8924560546875
        theta: 0.0
      size:
        x: 110.912109375
        y: 256.68017578125
    bbox3d:
      center:
        position:
          x: 0.0
          y: 0.0
          z: 0.0
        orientation:
          x: 0.0
          y: 0.0
          z: 0.0
          w: 1.0
      size:
        x: 0.0
        y: 0.0
        z: 0.0
      frame_id: camera_2/camera_link
    mask:
      height: 0
      width: 0
      data: []
    keypoints:
      data: []
    keypoints3d:
      data: []
      frame_id: camera_2/camera_link
  ---
```
When no detections are published by any of the cameras, the fusion node displays below empty header.
```shell
header:
  stamp:
    sec: 1760620602
    nanosec: 586551826
  frame_id: fused
detections:[]
```

## ROS2 Wrap for YOLO
ROS 2 wrap for YOLO models from [Ultralytics](https://github.com/ultralytics/ultralytics) to perform object detection and tracking, instance segmentation, human pose estimation and Oriented Bounding Box (OBB). There are also 3D versions of object detection, including instance segmentation, and human pose estimation based on depth images.

## Models

The compatible models for yolo_ros are the following:

- [YOLOv8](https://docs.ultralytics.com/models/yolov8/)


## Usage

<details>
<summary>Click to expand</summary>

### YOLOv8

```shell
ros2 launch yolo_bringup yolov8.launch.py
```

</details>

<p align="center">
  <img src="./docs/rqt_graph_yolov8.png" width="100%" />
</p>

### Topics

- **/yolo/detections**: Objects detected by YOLO using the RGB images. Each object contains a bounding box and a class name. It may also include a mark or a list of keypoints.
- **/yolo/tracking**: Objects detected and tracked from YOLO results. Each object is assigned a tracking ID.
- **/yolo/debug_image**: Debug images showing the detected and tracked objects. They can be visualized with rviz2.

### Parameters

These are the parameters from the [yolo.launch.py](./yolo_bringup/launch/yolo.launch.py), used to launch all models. Check out the [Ultralytics page](https://docs.ultralytics.com/modes/predict/#inference-arguments) for more details.

- **model_type**: Ultralytics model type (default: YOLO)
- **model**: YOLO model (default: yolov8m.pt)
- **tracker**: tracker file (default: bytetrack.yaml)
- **device**: GPU/CUDA (default: cuda:0)
- **enable**: whether to start YOLO enabled (default: True)
- **threshold**: detection threshold (default: 0.5)
- **iou**: intersection Over Union (IoU) threshold for Non-Maximum Suppression (NMS) (default: 0.7)
- **imgsz_height**: image height for inference (default: 480)
- **imgsz_width**: image width for inference (default: 640)
- **half**: whether to enable half-precision (FP16) inference speeding up model inference with minimal impact on accuracy (default: False)
- **max_det**: maximum number of detections allowed per image (default: 300)
- **augment**: whether to enable test-time augmentation (TTA) for predictions improving detection robustness at the cost of speed (default: False)
- **agnostic_nms**: whether to enable class-agnostic Non-Maximum Suppression (NMS) merging overlapping boxes of different classes (default: False)
- **retina_masks**: whether to use high-resolution segmentation masks if available in the model, enhancing mask quality for segmentation (default: False)
- **input_image_topic**: camera topic of RGB images (default: /camera/rgb/image_raw)
- **image_reliability**: reliability for the image topic: 0=system default, 1=Reliable, 2=Best Effort (default: 2)
- **input_depth_topic**: camera topic of depth images (default: /camera/depth/image_raw)
- **depth_image_reliability**: reliability for the depth image topic: 0=system default, 1=Reliable, 2=Best Effort (default: 2)
- **input_depth_info_topic**: camera topic for info data (default: /camera/depth/camera_info)
- **depth_info_reliability**: reliability for the depth info topic: 0=system default, 1=Reliable, 2=Best Effort (default: 2)
- **target_frame**: frame to transform the 3D boxes (default: base_link)
- **depth_image_units_divisor**: divisor to convert the depth image into meters (default: 1000)
- **maximum_detection_threshold**: maximum detection threshold in the z-axis (default: 0.3)
- **use_tracking**: whether to activate tracking after detection (default: True)
- **use_3d**: whether to activate 3D detections (default: False)
- **use_debug**: whether to activate debug node (default: True)

## Lifecycle Nodes

Previous updates add Lifecycle Nodes support to all the nodes available in the package.
This implementation tries to reduce the workload in the unconfigured and inactive states by only loading the models and activating the subscriber on the active state.

These are some resource comparisons using the default yolov8m.pt model on a 30fps video stream.

| State    | CPU Usage (i7 12th Gen) | VRAM Usage | Bandwidth Usage |
| -------- | ----------------------- | ---------- | --------------- |
| Active   | 40-50% in one core      | 628 MB     | Up to 200 Mbps  |
| Inactive | ~5-7% in one core       | 338 MB     | 0-20 Kbps       |

### YOLO 3D

```shell
ros2 launch yolo_bringup yolov8.launch.py use_3d:=True
```

<p align="center">
  <img src="./docs/rqt_graph_yolov8_3d.png" width="100%" />
</p>

## Demos

## Object Detection

This is the standard behavior of yolo_ros which includes object tracking.

```shell
ros2 launch yolo_bringup yolo.launch.py
```

[![](https://drive.google.com/thumbnail?authuser=0&sz=w1280&id=1gTQt6soSIq1g2QmK7locHDiZ-8MqVl2w)](https://drive.google.com/file/d/1gTQt6soSIq1g2QmK7locHDiZ-8MqVl2w/view?usp=sharing)

## Instance Segmentation

Instance masks are the borders of the detected objects, not all the pixels inside the masks.

```shell
ros2 launch yolo_bringup yolo.launch.py model:=yolov8m-seg.pt
```

[![](https://drive.google.com/thumbnail?authuser=0&sz=w1280&id=1dwArjDLSNkuOGIB0nSzZR6ABIOCJhAFq)](https://drive.google.com/file/d/1dwArjDLSNkuOGIB0nSzZR6ABIOCJhAFq/view?usp=sharing)

## Human Pose

Online persons are detected along with their keypoints.

```shell
ros2 launch yolo_bringup yolo.launch.py model:=yolov8m-pose.pt
```

[![](https://drive.google.com/thumbnail?authuser=0&sz=w1280&id=1pRy9lLSXiFEVFpcbesMCzmTMEoUXGWgr)](https://drive.google.com/file/d/1pRy9lLSXiFEVFpcbesMCzmTMEoUXGWgr/view?usp=sharing)

## 3D Object Detection

The 3D bounding boxes are calculated by filtering the depth image data from an RGB-D camera using the 2D bounding box. Only objects with a 3D bounding box are visualized in the 2D image.

```shell
ros2 launch yolo_bringup yolo.launch.py use_3d:=True
```

[![](https://drive.google.com/thumbnail?authuser=0&sz=w1280&id=1ZcN_u9RB9_JKq37mdtpzXx3b44tlU-pr)](https://drive.google.com/file/d/1ZcN_u9RB9_JKq37mdtpzXx3b44tlU-pr/view?usp=sharing)

## 3D Object Detection (Using Instance Segmentation Masks)

In this, the depth image data is filtered using the max and min values obtained from the instance masks. Only objects with a 3D bounding box are visualized in the 2D image.

```shell
ros2 launch yolo_bringup yolo.launch.py model:=yolov8m-seg.pt use_3d:=True
```

[![](https://drive.google.com/thumbnail?authuser=0&sz=w1280&id=1wVZgi5GLkAYxv3GmTxX5z-vB8RQdwqLP)](https://drive.google.com/file/d/1wVZgi5GLkAYxv3GmTxX5z-vB8RQdwqLP/view?usp=sharing)

## 3D Human Pose

Each keypoint is projected in the depth image and visualized using purple spheres. Only objects with a 3D bounding box are visualized in the 2D image.

```shell
ros2 launch yolo_bringup yolo.launch.py model:=yolov8m-pose.pt use_3d:=True
```

[![](https://drive.google.com/thumbnail?authuser=0&sz=w1280&id=1j4VjCAsOCx_mtM2KFPOLkpJogM0t227r)](https://drive.google.com/file/d/1j4VjCAsOCx_mtM2KFPOLkpJogM0t227r/view?usp=sharing)
