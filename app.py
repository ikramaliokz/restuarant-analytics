import cv2
import csv
import time
from ultralytics import YOLO, solutions
from datetime import datetime
from utils import  load_video, process_frame, display_analytics
from config import CONFIG


def main():
    detector_path = CONFIG['detector_path']
    classifier_path = CONFIG['classifier_path']
    video_path = CONFIG['video_path']
    
    detector = YOLO(detector_path)
    classifier = YOLO(classifier_path)

    heatmap_obj = solutions.Heatmap(
        colormap=cv2.COLORMAP_PARULA,
        view_img=False,
        shape="circle",
        classes_names=detector.names,
        decay_factor=1
    )
    
    cap = load_video(video_path)
    counted_ids_down, counted_ids_up = [], []
    tracked_boxes, tracked_classes = {}, {}
    counts = {'down': 0, 'up': 0, 'down_ids': counted_ids_down, 'up_ids': counted_ids_up}

    count_txt_color=(0, 0, 0)
    count_bg_color=(255, 255, 255)
    margin = 10

    width, height, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    video_writer = cv2.VideoWriter(CONFIG['output_video_path'], cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    print("hieght",height, "width", width)
    with open(CONFIG['result_csv'], 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Timestamp', 'Predicted Age', 'Predicted Gender'])

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame_time = time.time()
            frame = process_frame(frame, detector, classifier, tracked_boxes, counts, 
                                  CONFIG['x1_point'],CONFIG['x2_point'], CONFIG['y1_point'], CONFIG['y2_point'],
                                  tracked_classes, csv_writer, frame_time,
                                  heatmap_obj=heatmap_obj)

            counts_copy = counts.copy()
            counts_copy.pop('down_ids')
            counts_copy.pop('up_ids')

            display_analytics(frame, counts_copy, count_txt_color, count_bg_color, margin, left_align=True)
            display_analytics(frame, tracked_classes, count_txt_color, count_bg_color, margin)

            # text_point = (width - 250, 40)
            # for key, value in tracked_classes.items():
            #     result_string = f"{key}: {value} "
            #     cv2.putText(frame, result_string, text_point, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            #     text_point = (text_point[0], text_point[1] + 40)

            # cv2.putText(frame, f"Down: {counts['down']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, lineType=cv2.LINE_AA)
            # cv2.putText(frame, f"Up: {counts['up']}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, lineType=cv2.LINE_AA)
            video_writer.write(frame)
            cv2.imshow('Frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
