from ReceiveDetections import ReceiveDetectionsService
from ExtractingFeatures import ExtractingFeatures
from SendFeatures import SendFeatures
from CheckDetection import CheckDetection

import yaml
import threading
import argparse

from threading import Lock
from queue import Queue

def parse_args():
    parser = argparse.ArgumentParser()
    # Args concerning the establishment of crop zones for video 1 and video 2
    parser.add_argument('--input_conf', type=str, default='inputs_conf.yaml', help='Path to the input configuration file, that contains crop zones and mqtt topics for receiving info from each camera')
    parser.add_argument('--mqtt_broker', type=str, default='localhost', help='Address of the MQTT broker')
    parser.add_argument('--mqtt_port', type=int, default=1884, help='Port of the MQTT broker')

    return parser.parse_args()

# Shared queue for inference requests
inference_queue = Queue()

def inference_worker(extractor, extractor_lock):
    """Single thread that owns the CUDA context and runs inference."""
    while True:
        track_id, image, sender = inference_queue.get()
        if track_id is None:  # poison pill for clean shutdown
            break
        with extractor_lock:
            features = extractor.get_feature(image)
        sender(track_id, features)


# ----------------------- Te es beidzu ---------------------------------------

def process_stream(cam_name, cam_params, extractor, extractor_lock, args):

    print(f"Starting processing thread for {cam_name}")

    receiver = ReceiveDetectionsService(broker=args.mqtt_broker, port=args.mqtt_port, topic=cam_params["mqtt_topic"])
    mode = cam_params["mode"]
    check = CheckDetection(
        cam_params["crop_zone_rows"],
        cam_params["crop_zone_cols"],
        tuple(cam_params["crop_zone_area_bottom_left"]),
        tuple(cam_params["crop_zone_area_top_right"])
    )

    if mode == 'comp':
        sender = SendFeatures(mqtt_broker= args.mqtt_broker, mqtt_port=args.mqtt_port, mqtt_topic="tomass/compare_features")
    else:
        sender = SendFeatures(mqtt_broker= args.mqtt_broker, mqtt_port=args.mqtt_port, mqtt_topic="tomass/save_features")

    while True:
        new_images = receiver.get_pending_images()
        for entry in new_images:
            image = entry["image"]
            track_id = entry["track_id"]
            bbox = entry["bbox"]

            if check.perform_checks(track_id, bbox):
                # Extract features and use extractor lock to ensure thread safety. That way only one thread hits the GPU at a time.
                with extractor_lock:
                    # Use the shared extractor
                    features = extractor.get_feature(image)
                sender(track_id, features)

def main(cons_args):

    # Load config file
    with open(cons_args.input_conf, "r") as f:
        config = yaml.safe_load(f)

    extractor = ExtractingFeatures()

    extractor_lock = Lock()

    threads = []

    for cam_name, cam_params in config["streams"].items():
        t = threading.Thread(target=process_stream, args=(cam_name, cam_params, extractor, extractor_lock, cons_args), name=cam_name)
        t.start()
        threads.append(t)

    
    for t in threads:
        t.join()

if __name__ == "__main__":
    args = parse_args()
    main(args)

