#!/usr/bin/env python3
"""
yolo_fusion_node.py (updated: supports 1..6 cameras, 2x3 image stitch)

Behavior summary:
 - Detections (append-only): only accept detections that arrived *since the last publish*.
   If none arrived in the cycle, publish an empty yolo_msgs/DetectionArray with header.frame_id='fused'.
 - Images: keep last image as before unless the publisher for that camera is gone (publisher count == 0).
   If publisher gone -> show black tile for that position.
 - Preserves per-detection bbox3d.frame_id / keypoints3d.frame_id to record source camera.
 - Uses cv_bridge + OpenCV for image conversions. Stitching forms a 2x3 grid (left->right, top->bottom)
   and tile placement follows the order of img_input_topics.
 - Accepts between 1 and 6 detection topics and 1 and 6 image topics. Missing grid cells are rendered black.
"""

from __future__ import annotations
import argparse
from copy import deepcopy
import sys
import traceback
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
import numpy as np

from builtin_interfaces.msg import Time
from sensor_msgs.msg import Image as RosImage
from yolo_msgs.msg import DetectionArray

# OpenCV + cv_bridge
try:
    import cv2
    from cv_bridge import CvBridge, CvBridgeError
except Exception:
    cv2 = None
    CvBridge = None
    CvBridgeError = Exception  # fallback

# Defaults updated to support up to 6 topics (grid is 2x3)
DEFAULT_DET_INPUTS = [
    '/yolo/detections1',
    '/yolo/detections2',
    '/yolo/detections3',
    '/yolo/detections4',
    '/yolo/detections5',
    '/yolo/detections6',   # optional: uncomment or pass via CLI
]
DEFAULT_IMG_INPUTS = [
    '/yolo/dbg_image3',
    '/yolo/dbg_image1',
    '/yolo/dbg_image2',
    '/yolo/dbg_image5',
    '/yolo/dbg_image6',    # optional: uncomment or pass via CLI
    '/yolo/dbg_image4',
]
DEFAULT_OUTPUT_DET = '/fused/detections'
DEFAULT_OUTPUT_IMG = '/fused/dbg_image'
DEFAULT_RATE = 10.0
NEUTRAL_FUSED_FRAME_ID = 'fused'  # fused header.frame_id

# Grid configuration for image stitching
GRID_ROWS = 2
GRID_COLS = 3
GRID_CELLS = GRID_ROWS * GRID_COLS  # 6 cells for 2x3

class YoloFusionNode(Node):
    def __init__(self,
                 det_input_topics: list[str],
                 img_input_topics: list[str],
                 output_det_topic: str,
                 output_img_topic: str,
                 publish_rate_hz: float):
        super().__init__('late_fusion_node')

        cam_count = max(len(det_input_topics), len(img_input_topics))
        self.get_logger().info(f'YOLO fusion node starting (freshness-safe, supports 1..{GRID_CELLS} cams). Using up to {cam_count} cameras.')

        self.det_input_topics = det_input_topics
        self.img_input_topics = img_input_topics
        self.output_det_topic = output_det_topic
        self.output_img_topic = output_img_topic
        self.publish_rate_hz = float(publish_rate_hz)

        qos = QoSProfile(depth=10)

        # Latest messages storage
        self.latest_det_msgs: dict[str, DetectionArray | None] = {t: None for t in self.det_input_topics}
        self.recv_seq: dict[str, int] = {t: 0 for t in self.det_input_topics}           # incremented on each msg recv
        self.last_published_seq: dict[str, int] = {t: 0 for t in self.det_input_topics}  # snapshot after publish

        # For images we keep latest message and will display last image unless publisher disappears
        self.latest_img_msgs: dict[str, RosImage | None] = {t: None for t in self.img_input_topics}

        # CV bridge
        self.bridge = CvBridge() if CvBridge is not None else None
        if self.bridge is None:
            self.get_logger().warn('cv_bridge not available. Image stitching will not work until cv_bridge + OpenCV installed.')

        # Subscriptions: detections
        for topic in self.det_input_topics:
            try:
                self.create_subscription(
                    DetectionArray,
                    topic,
                    lambda msg, topic=topic: self._det_callback(msg, topic),
                    qos
                )
                self.get_logger().info(f'Subscribed to detection topic: {topic}')
            except Exception as e:
                self.get_logger().error(f'Failed to subscribe to detection topic {topic}: {e}')

        # Subscriptions: images
        for topic in self.img_input_topics:
            try:
                self.create_subscription(
                    RosImage,
                    topic,
                    lambda msg, topic=topic: self._img_callback(msg, topic),
                    qos
                )
                self.get_logger().info(f'Subscribed to image topic: {topic}')
            except Exception as e:
                self.get_logger().error(f'Failed to subscribe to image topic {topic}: {e}')

        # Publishers
        self.det_pub = self.create_publisher(DetectionArray, self.output_det_topic, qos)
        self.img_pub = self.create_publisher(RosImage, self.output_img_topic, qos)

        # Timer
        self.timer = self.create_timer(1.0 / float(self.publish_rate_hz), self._on_timer)
        self.get_logger().info(f'Publishing fused detections to {self.output_det_topic} and stitched image to {self.output_img_topic} @ {self.publish_rate_hz} Hz')

    # Detections callback: store message and increment seq
    def _det_callback(self, msg: DetectionArray, topic: str) -> None:
        try:
            self.latest_det_msgs[topic] = deepcopy(msg)
        except Exception:
            self.latest_det_msgs[topic] = msg
        # mark that this topic has a new message
        self.recv_seq[topic] += 1

    # Image callback: keep latest (we'll decide use vs black based on publisher existence)
    def _img_callback(self, msg: RosImage, topic: str) -> None:
        try:
            self.latest_img_msgs[topic] = deepcopy(msg)
        except Exception:
            self.latest_img_msgs[topic] = msg

    # Timer loop: produce fused detections and stitched image
    def _on_timer(self) -> None:
        try:
            self._publish_fused_detections()
            self._publish_stitched_image()
        except Exception as e:
            self.get_logger().error(f'Error in timer loop: {e}\n{traceback.format_exc()}')

    # -------------------------
    # Detections: freshness-enforced append-only
    # -------------------------
    def _publish_fused_detections(self) -> None:
        # Collect only messages that are NEW since last publish (recv_seq > last_published_seq)
        msgs_to_use = []
        for topic in self.det_input_topics:
            if self.recv_seq.get(topic, 0) > self.last_published_seq.get(topic, 0):
                msg = self.latest_det_msgs.get(topic)
                if msg is not None:
                    msgs_to_use.append((topic, msg))

        # If no new messages across all detection topics -> publish empty fused message
        if not msgs_to_use:
            empty_msg = DetectionArray()
            now = self.get_clock().now().to_msg()
            empty_msg.header.stamp = deepcopy(now)
            empty_msg.header.frame_id = NEUTRAL_FUSED_FRAME_ID
            empty_msg.detections = []
            try:
                self.det_pub.publish(empty_msg)
                self.get_logger().debug('Published EMPTY fused detections (no fresh inputs this cycle)')
            except Exception as e:
                self.get_logger().error(f'Failed to publish empty fused detections: {e}')
            # Advance last_published_seq to current recv_seq for all topics so old messages won't be reused
            for t in self.det_input_topics:
                self.last_published_seq[t] = self.recv_seq.get(t, 0)
            return

        # There are new messages; build fused message from only those new messages
        try:
            newest = max((m for (_, m) in msgs_to_use), key=lambda m: (m.header.stamp.sec, m.header.stamp.nanosec))
        except Exception:
            newest = msgs_to_use[0][1]

        out_msg: DetectionArray = deepcopy(newest)
        out_msg.detections = []

        for topic, parent_msg in msgs_to_use:
            for det in parent_msg.detections:
                det_copy = deepcopy(det)
                # record source camera frame into bbox3d.frame_id and keypoints3d.frame_id if present
                try:
                    if hasattr(det_copy, 'bbox3d'):
                        try:
                            det_copy.bbox3d.frame_id = parent_msg.header.frame_id
                        except Exception:
                            pass
                except Exception:
                    pass
                try:
                    if hasattr(det_copy, 'keypoints3d'):
                        try:
                            det_copy.keypoints3d.frame_id = parent_msg.header.frame_id
                        except Exception:
                            pass
                except Exception:
                    pass
                out_msg.detections.append(det_copy)

        # set header (stamp = newest among used msgs)
        newest_stamp = Time()
        newest_stamp.sec = 0
        newest_stamp.nanosec = 0
        for _, m in msgs_to_use:
            try:
                s = m.header.stamp
                if (s.sec, s.nanosec) > (newest_stamp.sec, newest_stamp.nanosec):
                    newest_stamp = deepcopy(s)
            except Exception:
                pass
        out_msg.header.stamp = newest_stamp
        out_msg.header.frame_id = NEUTRAL_FUSED_FRAME_ID

        # Publish fused detections
        try:
            self.det_pub.publish(out_msg)
            self.get_logger().debug(f'Published fused detections ({len(out_msg.detections)} detections) from fresh inputs')
        except Exception as e:
            self.get_logger().error(f'Failed to publish fused detections: {e}')

        # After publishing, mark last_published_seq = current recv_seq for those topics consumed (and for all topics to prevent reuse)
        for t in self.det_input_topics:
            self.last_published_seq[t] = self.recv_seq.get(t, 0)

    # -------------------------
    # Images: 2x3 grid stitching, tile order = img_input_topics order
    # -------------------------
    def _publish_stitched_image(self) -> None:
        # If cv_bridge/OpenCV missing, do nothing
        if self.bridge is None or cv2 is None:
            return

        # Build images list mapped to GRID_CELLS positions: fill with None when no topic assigned or publisher disconnected
        imgs = []
        img_msgs = []

        # For each grid cell index, map to input topic if available (order defines position)
        for idx in range(GRID_CELLS):
            if idx < len(self.img_input_topics):
                topic = self.img_input_topics[idx]
                msg = self.latest_img_msgs.get(topic)
                # check whether publisher exists for this topic; if not -> treat as disconnected -> force black
                pubs = self.get_publishers_info_by_topic(topic)
                publisher_count = len(pubs) if pubs is not None else 0
                publisher_connected = (publisher_count > 0)
                if msg is None:
                    # no image ever received yet
                    if publisher_connected:
                        imgs.append(None)
                        img_msgs.append(None)
                    else:
                        imgs.append(None)
                        img_msgs.append(None)
                    continue

                if not publisher_connected:
                    imgs.append(None)
                    img_msgs.append(None)
                    continue

                # convert to cv image
                try:
                    cv_img = None
                    try:
                        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                    except CvBridgeError:
                        try:
                            cv_img_raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                            if cv_img_raw is None:
                                cv_img = None
                            else:
                                if len(cv_img_raw.shape) == 2:
                                    cv_img = cv2.cvtColor(cv_img_raw, cv2.COLOR_GRAY2BGR)
                                elif cv_img_raw.shape[2] == 3:
                                    cv_img = cv2.cvtColor(cv_img_raw, cv2.COLOR_RGB2BGR)
                                else:
                                    cv_img = cv_img_raw
                        except Exception as e:
                            self.get_logger().warning(f'cv_bridge passthrough conversion failed for topic {topic}: {e}')
                            cv_img = None

                    if cv_img is None:
                        imgs.append(None)
                        img_msgs.append(msg)
                        continue

                    imgs.append(cv_img)
                    img_msgs.append(msg)
                except Exception as e:
                    self.get_logger().warning(f'Exception converting image from {topic}: {e}')
                    imgs.append(None)
                    img_msgs.append(msg)
            else:
                # No input topic for this grid cell -> black placeholder
                imgs.append(None)
                img_msgs.append(None)

        # Determine target tile size
        available_cv = [im for im in imgs if im is not None]
        if available_cv:
            heights = [im.shape[0] for im in available_cv if im is not None and im.shape[0] > 0]
            target_h = min(heights) if heights else 240
            widths = [im.shape[1] for im in available_cv if im is not None and im.shape[1] > 0]
            median_w = int(sorted(widths)[len(widths)//2]) if widths else 320
        else:
            target_h = 240
            median_w = 320

        # Prepare tiles: each tile => exact (target_h, median_w, 3) image
        prepared = []
        for im in imgs:
            if im is None:
                black = np.zeros((target_h, median_w, 3), dtype=np.uint8)
                prepared.append(black)
                continue
            try:
                h, w = im.shape[0], im.shape[1]
                # Resize preserving aspect ratio then final-resize to exact median_w x target_h to keep grid aligned
                if h != target_h:
                    scale = target_h / float(h)
                    new_w = max(1, int(round(w * scale)))
                    try:
                        im_resized = cv2.resize(im, (new_w, target_h), interpolation=cv2.INTER_AREA)
                    except Exception:
                        im_resized = cv2.resize(im, (new_w, target_h))
                else:
                    im_resized = im
                # Now force final width = median_w (may warp slightly but ensures tidy grid)
                if im_resized.shape[1] != median_w:
                    try:
                        im_final = cv2.resize(im_resized, (median_w, target_h), interpolation=cv2.INTER_AREA)
                    except Exception:
                        im_final = cv2.resize(im_resized, (median_w, target_h))
                else:
                    im_final = im_resized

                # Ensure 3 channels and uint8
                if len(im_final.shape) == 2:
                    im_final = cv2.cvtColor(im_final, cv2.COLOR_GRAY2BGR)
                elif im_final.shape[2] == 4:
                    im_final = cv2.cvtColor(im_final, cv2.COLOR_BGRA2BGR)
                if im_final.dtype != np.uint8:
                    im_final = im_final.astype(np.uint8)

                prepared.append(im_final)
            except Exception as e:
                self.get_logger().warning(f'Failed preparing tile image: {e}')
                black = np.zeros((target_h, median_w, 3), dtype=np.uint8)
                prepared.append(black)

        # Build each row by horizontally concatenating the row tiles, then vertically stack rows
        row_images = []
        try:
            for r in range(GRID_ROWS):
                start = r * GRID_COLS
                end = start + GRID_COLS
                row_tiles = prepared[start:end]
                try:
                    row_concat = cv2.hconcat(row_tiles)
                except Exception:
                    # fallback: ensure dtype uint8 and try again
                    row_tiles2 = [ (p.astype('uint8') if p.dtype != np.uint8 else p) for p in row_tiles ]
                    row_concat = cv2.hconcat(row_tiles2)
                row_images.append(row_concat)

            try:
                panorama = cv2.vconcat(row_images)
            except Exception:
                # fallback: try converting dtypes then vconcat
                row_images2 = [ (rimg.astype('uint8') if rimg.dtype != np.uint8 else rimg) for rimg in row_images ]
                panorama = cv2.vconcat(row_images2)
        except Exception as e:
            self.get_logger().error(f'Failed to build panorama: {e}')
            return

        # Convert to ROS Image
        try:
            ros_img = self.bridge.cv2_to_imgmsg(panorama, encoding='bgr8')
        except Exception:
            try:
                ros_img = self.bridge.cv2_to_imgmsg(cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB), encoding='rgb8')
            except Exception as e:
                self.get_logger().error(f'cv_bridge failed to convert panorama to Image: {e}')
                return

        # Header stamp: use current time (safe)
        newest_stamp = self.get_clock().now().to_msg()
        ros_img.header.stamp = deepcopy(newest_stamp)
        ros_img.header.frame_id = NEUTRAL_FUSED_FRAME_ID

        # Publish stitched image
        try:
            self.img_pub.publish(ros_img)
            self.get_logger().debug('Published fused dbg image panorama (2x3)')
        except Exception as e:
            self.get_logger().error(f'Failed to publish fused dbg image: {e}')

def parse_args(argv):
    parser = argparse.ArgumentParser(description=f'YOLO fusion node with image stitching (1..{GRID_CELLS} cams, 2x3 grid)')
    parser.add_argument('--det_inputs', nargs='*',
                        help=f'Detection input topics (1..{GRID_CELLS}). If omitted, defaults are used.',
                        default=DEFAULT_DET_INPUTS)
    parser.add_argument('--img_inputs', nargs='*',
                        help=f'Debug image input topics (1..{GRID_CELLS}). Order defines tile positions left->right, top->bottom.',
                        default=DEFAULT_IMG_INPUTS)
    parser.add_argument('--output_det', default=DEFAULT_OUTPUT_DET,
                        help='Fused detection output topic (default: /fused/detections)')
    parser.add_argument('--output_img', default=DEFAULT_OUTPUT_IMG,
                        help='Fused debug image output topic (default: /fused/dbg_image)')
    parser.add_argument('--rate', '-r', type=float, default=DEFAULT_RATE,
                        help='Publish rate in Hz (default: 10)')
    return parser.parse_args(argv)

def main(argv=None):
    args = parse_args(argv if argv is not None else sys.argv[1:])

    det_inputs = args.det_inputs
    img_inputs = args.img_inputs
    # normalize defaults that might include commented-out placeholders
    det_inputs = [t for t in det_inputs if t is not None and t != '']
    img_inputs = [t for t in img_inputs if t is not None and t != '']

    if not (1 <= len(det_inputs) <= GRID_CELLS) or not (1 <= len(img_inputs) <= GRID_CELLS):
        print(f'ERROR: provide between 1 and {GRID_CELLS} detection topics and image topics', file=sys.stderr)
        return 2

    rclpy.init()
    node = None
    try:
        node = YoloFusionNode(det_inputs, img_inputs, args.output_det, args.output_img, args.rate)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Node crashed: {e}\n{traceback.format_exc()}', file=sys.stderr)
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
