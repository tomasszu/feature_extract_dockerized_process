from ReceiveDetections import ReceiveDetectionsService
from ExtractingFeatures import ExtractingFeatures
from SendFeatures import SendFeatures
from CheckDetection import CheckDetection

# In your main loop (e.g., every N ms)
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['save', 'comp'], default='save')

    # Args concerning the establishment of crop zones for video 1 and video 2
    parser.add_argument('--crop_zone_rows', type=int, default=7, help='Number of rows in the crop zone grid for the video.')
    parser.add_argument('--crop_zone_cols', type=int, default=6, help='Number of columns in the crop zone grid for the video.')
    parser.add_argument('--crop_zone_area_bottom_left', type=tuple, default=(0, 1000), help='Bottom-left corner of the crop zone area as a tuple (x, y) for the video.')
    parser.add_argument('--crop_zone_area_top_right', type=tuple, default=(1750, 320), help='Top-right corner of the crop zone area as a tuple (x, y) for the video.')
    
    return parser.parse_args()

def main(args):

    receiver = ReceiveDetectionsService()
    extractor = ExtractingFeatures()
    check = CheckDetection(args.crop_zone_rows, args.crop_zone_cols,
                           args.crop_zone_area_bottom_left, args.crop_zone_area_top_right)

    if(args.mode == 'comp'):
        sender = SendFeatures(mqtt_topic="tomass/compare_features")
    else:
        sender = SendFeatures(mqtt_topic="tomass/save_features")


    while True:
        new_images = receiver.get_pending_images()
        for entry in new_images:

            image = entry["image"]
            track_id = entry["track_id"]
            bbox = entry["bbox"]
            #print(f"Ready for feature extraction: Track {track_id}, Shape: {image.shape}")

            if check.perform_checks(track_id, bbox):

                features = extractor.get_feature(image)

                sender(track_id, features)
            
        #time.sleep(0.1)


if __name__ == "__main__":
    args = parse_args()
    main(args)

