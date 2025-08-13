import json
import cv2
import numpy as np
import paho.mqtt.client as mqtt
from PIL import Image
import time

class ReceiveDetectionsService:
    def __init__(self, broker="localhost", port=1884, topic="tomass/detections"):
        self.topic = topic
        self.queue = []  # stores (track_id, bbox, timestamp, image_np)
        self.client = mqtt.Client(client_id="receiver1")
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.connected = False

        self.client.connect(broker, port, 60)
        self.client.loop_start()  # non-blocking

        # Wait until connection is established
        timeout = time.time() + 5  # max 5 seconds wait
        while not self.connected:
            if time.time() > timeout:
                raise TimeoutError("MQTT connection timed out.")
            time.sleep(0.5)

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print("[MQTT] Connected successfully")
            client.subscribe(self.topic)  # ← THIS WAS MISSING
            self.connected = True
        else:
            print(f"[MQTT] Connection failed with code {rc}")

    def _on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode('utf-8'))

            image_np = self._decode_crop_pil(payload["image"])
            self.queue.append({
                "track_id": payload["track_id"],
                "bbox": payload["bbox"],
                "image": image_np
            })
            print(f"[MQTT] Received and decoded crop for track_id {payload['track_id']}")
        except Exception as e:
            print(f"[ERROR] Failed to process MQTT message: {e}")

    def _decode_crop_np(self, encoded_crop):
        crop_bytes = bytes.fromhex(encoded_crop)
        np_arr = np.frombuffer(crop_bytes, dtype=np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode image from base64")
        return image
    
    def _decode_crop_pil(self, encoded_crop):
        crop_bytes = bytes.fromhex(encoded_crop)
        np_arr = np.frombuffer(crop_bytes, dtype=np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Failed to decode image from PNG bytes")

        # Convert OpenCV BGR → PIL RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(image_rgb)

    def get_pending_images(self):
        """Retrieve and clear the queue of received crops"""
        data = self.queue[:]
        self.queue.clear()
        return data
