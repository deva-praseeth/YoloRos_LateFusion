#!/usr/bin/env python3
"""
yolo_fusion_node.py (updated: strict detection freshness + image black-on-disconnect)

Behavior summary:
 - Detections (append-only): only accept detections that arrived *since the last publish*.
   If none arrived in the cycle, publish an empty yolo_msgs/DetectionArray with header.frame_id='fused'.
 - Images: keep last image as before unless the publisher for that camera is gone (publisher count == 0).
   If publisher gone -> show black tile for that column.
 - Preserves per-detection bbox3d.frame_id / keypoints3d.frame_id to record source camera.
 - Uses cv_bridge + OpenCV for image conversions (stitching is side-by-side).
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

DEFAULT_DET_INPUTS = [
    '/yolo_cam2/detections',
    '/yolo_cam3/detections',
    '/yolo_cam4/detections',
]
DEFAULT_IMG_INPUTS = [
    '/yolo_cam2/dbg_image',
    '/yolo_cam3/dbg_image',
    '/yolo_cam4/dbg_image',
]
DEFAULT_OUTPUT_DET = '/fused/detections'
DEFAULT_OUTPUT_IMG = '/fused/dbg_image'
DEFAULT_RATE = 10.0
NEUTRAL_FUSED_FRAME_ID = 'fused'  # fused header.frame_id

class YoloFusionNode(Node):
    def __init__(self,
                 det_input_topics: list[str],
                 img_input_topics: list[str],
                 output_det_topic: str,
                 output_img_topic: str,
                 publish_rate_hz: float):
        super().__init__('late_fusion_node')

        self.get_logger().info('YOLO fusion node starting (freshness-safe).')

        self.det_input_topics = det_input_topics
        self.img_input_topics = img_input_topics
        self.output_det_topic = output_det_topic
        self.output_img_topic = output_img_topic
        self.publish_rate_hz = float(publish_rate_hz)

        qos = QoSProfile(depth=10)

        # Latest messages storage
        # For detections we track a receive counter per-topic so we can decide "fresh since last publish"
        self.latest_det_msgs: dict[str, DetectionArray | None] = {t: None for t in self.det_input_topics}
        self.recv_seq: dict[str, int] = {t: 0 for t in self.det_input_topics}           # incremented on each msg recv
        self.last_published_seq: dict[str, int] = {t: 0 for t in self.det_input_topics}  # snapshot after publish

        # For images we keep latest message and will display last image unless publisher disappears
        self.latest_img_msgs: dict[str, RosImage | None] = {t: None for t in self.img_input_topics}
        # don't track seq for images (we keep last image unless publisher count == 0)

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
            # stamp with current ROS time (fresh), and neutral fused frame_id
            now = self.get_clock().now().to_msg()
            empty_msg.header.stamp = deepcopy(now)
            empty_msg.header.frame_id = NEUTRAL_FUSED_FRAME_ID
            empty_msg.detections = []
            try:
                self.det_pub.publish(empty_msg)
                self.get_logger().debug('Published EMPTY fused detections (no fresh inputs this cycle)')
            except Exception as e:
                self.get_logger().error(f'Failed to publish empty fused detections: {e}')
            # Important: advance last_published_seq to current recv_seq for all topics so old messages won't be reused
            for t in self.det_input_topics:
                self.last_published_seq[t] = self.recv_seq.get(t, 0)
            return

        # There are new messages; build fused message from only those new messages
        # Use the newest stamp among the msgs_to_use as template/time
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
    # Images: keep last image unless publisher disappears -> black placeholder
    # -------------------------
    def _publish_stitched_image(self) -> None:
        # If cv_bridge/OpenCV missing, do nothing
        if self.bridge is None or cv2 is None:
            return

        # Build images list in order of img_input_topics
        imgs = []
        img_msgs = []
        for topic in self.img_input_topics:
            msg = self.latest_img_msgs.get(topic)
            # check whether publisher exists for this topic; if not -> treat as disconnected -> force black
            pubs = self.get_publishers_info_by_topic(topic)
            publisher_count = len(pubs) if pubs is not None else 0
            publisher_connected = (publisher_count > 0)
            if msg is None:
                # no image ever received yet
                if publisher_connected:
                    # publisher exists but not yet produced frame -> treat as missing this cycle: preserve None (will create black placeholder)
                    imgs.append(None)
                    img_msgs.append(None)
                else:
                    # no publisher -> black
                    imgs.append(None)
                    img_msgs.append(None)
                continue

            # If publisher exists -> we show last image (previous behavior)
            # If publisher gone -> black (we check publisher_connected)
            if not publisher_connected:
                imgs.append(None)  # force black tile
                img_msgs.append(None)
                continue

            # else publisher exists and msg present: convert to cv image
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

        # If no images at all available and no publishers -> publish a single black panorama
        available_cv = [im for im in imgs if im is not None]
        # choose target height = min of available heights if any, else 240 as fallback
        if available_cv:
            heights = [im.shape[0] for im in available_cv if im is not None and im.shape[0] > 0]
            target_h = min(heights) if heights else 240
            widths = [im.shape[1] for im in available_cv if im is not None and im.shape[1] > 0]
            median_w = int(sorted(widths)[len(widths)//2]) if widths else 320
        else:
            # No images available (no publisher or none sent frames) -> produce all-black panorama
            target_h = 240
            median_w = 320

        # Prepare tiles: if img None -> black placeholder; else resize to target_h
        prepared = []
        for im in imgs:
            if im is None:
                black = np.zeros((target_h, median_w, 3), dtype=np.uint8)
                prepared.append(black)
            else:
                h, w = im.shape[0], im.shape[1]
                if h != target_h:
                    scale = target_h / float(h)
                    new_w = max(1, int(round(w * scale)))
                    try:
                        im_resized = cv2.resize(im, (new_w, target_h), interpolation=cv2.INTER_AREA)
                    except Exception:
                        im_resized = cv2.resize(im, (new_w, target_h))
                else:
                    im_resized = im
                prepared.append(im_resized)

        # Normalize channels & dtype
        for i, p in enumerate(prepared):
            if p is None:
                prepared[i] = np.zeros((target_h, median_w, 3), dtype=np.uint8)
            else:
                if len(p.shape) == 2:
                    prepared[i] = cv2.cvtColor(p, cv2.COLOR_GRAY2BGR)
                elif p.shape[2] == 4:
                    prepared[i] = cv2.cvtColor(p, cv2.COLOR_BGRA2BGR)
                if prepared[i].dtype != np.uint8:
                    prepared[i] = prepared[i].astype(np.uint8)

        # Concatenate horizontally
        try:
            panorama = cv2.hconcat(prepared)
        except Exception:
            try:
                prepared2 = [ (p.astype('uint8') if p.dtype != np.uint8 else p) for p in prepared ]
                panorama = cv2.hconcat(prepared2)
            except Exception as e:
                self.get_logger().error(f'Failed to horizontally concatenate images: {e}')
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

        # Header stamp: prefer newest detection stamp (if any fresh detections were used last cycle),
        # else use newest available image stamp, else current time.
        newest_stamp = None
        # check detection messages used recently (we can use latest_det_msgs but they may be stale; choose current time)
        # For simplicity, set now as stamp
        newest_stamp = self.get_clock().now().to_msg()

        ros_img.header.stamp = deepcopy(newest_stamp)
        ros_img.header.frame_id = NEUTRAL_FUSED_FRAME_ID

        # Publish stitched image
        try:
            self.img_pub.publish(ros_img)
            self.get_logger().debug('Published fused dbg image panorama')
        except Exception as e:
            self.get_logger().error(f'Failed to publish fused dbg image: {e}')

def parse_args(argv):
    parser = argparse.ArgumentParser(description='YOLO fusion node with image stitching (single file)')
    parser.add_argument('--det_inputs', nargs=3, metavar=('D1','D2','D3'),
                        help='Three detection input topics (default cam2,cam3,cam4).',
                        default=DEFAULT_DET_INPUTS)
    parser.add_argument('--img_inputs', nargs=3, metavar=('I1','I2','I3'),
                        help='Three debug image input topics (default cam2,cam3,cam4).',
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
    if len(det_inputs) != 3 or len(img_inputs) != 3:
        print('ERROR: provide exactly 3 detection topics and 3 image topics', file=sys.stderr)
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
