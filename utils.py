import cv2
from datetime import datetime
from PIL import Image

def load_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Error reading video file: {video_path}")
    return cap

def xywh_to_xyxy(x, y, w, h):
    x_min = x - w / 2
    y_min = y - h / 2
    x_max = x + w / 2
    y_max = y + h / 2
    return int(x_min), int(y_min), int(x_max), int(y_max)

def classifier_call(box, frame, classifier):
    x_min, y_min, x_max, y_max = xywh_to_xyxy(box[0], box[1], box[2], box[3])
    box_crop = frame[y_min:y_max, x_min:x_max]
    # box_crop_img = Image.fromarray(box_crop)
    # print(box_crop_img.shape, type(box_crop_img))
    classification_result = classifier.predict(box_crop)
    class_names_dict = classification_result[0].names
    class_with_max_prob = classification_result[0].probs.top1
    return class_names_dict[class_with_max_prob]

# def display_analytics(im0, text, txt_color, bg_color, margin, alpha=0.4, left_align=False):
#     """
#     Display the overall statistics for parking lots with semi-transparent background.
#     Args:
#         im0 (ndarray): inference image
#         text (dict): labels dictionary
#         txt_color (bgr color): display color for text foreground
#         bg_color (bgr color): display color for text background
#         margin (int): gap between text and rectangle for better display
#         alpha (float): opacity level of the background rectangle
#         left_align (bool): flag to align text on the left
#     """
#     lw = max(round(sum(im0.shape) / 2 * 0.003), 2)
#     tf = max(lw - 1, 1)
#     sf = lw / 3
#     horizontal_gap = int(im0.shape[1] * 0.02)
#     vertical_gap = int(im0.shape[0] * 0.01)
#     text_y_offset = 0

#     for label, value in text.items():
#         txt = f"{label}: {value}"
#         text_size = cv2.getTextSize(txt, 0, sf, tf)[0]
#         if text_size[0] < 5 or text_size[1] < 5:
#             text_size = (5, 5)
#         text_x = im0.shape[1] - text_size[0] - margin * 2 - horizontal_gap
#         text_y = text_y_offset + text_size[1] + margin * 2 + vertical_gap
#         rect_x1 = text_x - margin * 2
#         rect_y1 = text_y - text_size[1] - margin * 2
#         rect_x2 = text_x + text_size[0] + margin * 2
#         rect_y2 = text_y + margin * 2

#         if left_align:
#             text_x = margin
#             rect_x1 = 0
#             rect_x2 = text_size[0] + margin * 2

#         # Create a copy for the area to draw the rectangle
#         sub_img = im0[rect_y1:rect_y2, rect_x1:rect_x2].copy()
#         sub_img = cv2.rectangle(sub_img, (0, 0), (sub_img.shape[1], sub_img.shape[0]), bg_color, -1)
#         im0[rect_y1:rect_y2, rect_x1:rect_x2] = cv2.addWeighted(sub_img, alpha, im0[rect_y1:rect_y2, rect_x1:rect_x2], 1 - alpha, 0)

#         # Put text on the image
#         cv2.putText(im0, txt, (text_x, text_y), 0, sf, txt_color, tf, lineType=cv2.LINE_AA)
#         text_y_offset = rect_y2

def display_analytics(im0, text, txt_color, bg_color, margin, alpha=0.4, left_align=False):
    """
    Display the overall statistics for parking lots with semi-transparent background.
    Args:
        im0 (ndarray): inference image
        text (dict): labels dictionary
        txt_color (bgr color): display color for text foreground
        bg_color (bgr color): display color for text background
        margin (int): gap between text and rectangle for better display
        alpha (float): opacity level of the background rectangle
        left_align (bool): flag to align text on the left
    """
    lw = max(round(sum(im0.shape) / 2 * 0.003), 2)
    tf = max(lw - 1, 1)
    sf = lw / 3
    horizontal_gap = int(im0.shape[1] * 0.02)
    vertical_gap = int(im0.shape[0] * 0.01)
    text_y_offset = 0

    max_text_size = (0, 0)

    for label, value in text.items():
        txt = f"{label}: {value}"
        text_size = cv2.getTextSize(txt, 0, sf, tf)[0]
        if text_size[0] > max_text_size[0]:
            max_text_size = text_size

    text_size = max_text_size

    if text_size[0] < 5 or text_size[1] < 5:
        text_size = (5, 5)
    
    for label, value in text.items():
        txt = f"{label}: {value}"
        text_x = im0.shape[1] - text_size[0] - margin * 2 - horizontal_gap
        text_y = text_y_offset + text_size[1] + margin * 2 + vertical_gap
        rect_x1 = text_x - margin * 2
        rect_y1 = text_y - text_size[1] - margin * 2
        rect_x2 = text_x + text_size[0] + margin * 2
        rect_y2 = text_y + margin * 2

        if left_align:
            text_x = margin
            rect_x1 = 0
            rect_x2 = text_size[0] + margin * 2

        # Create a copy for the area to draw the rectangle
        sub_img = im0[rect_y1:rect_y2, rect_x1:rect_x2].copy()
        sub_img = cv2.rectangle(sub_img, (0, 0), (sub_img.shape[1], sub_img.shape[0]), bg_color, -1)
        im0[rect_y1:rect_y2, rect_x1:rect_x2] = cv2.addWeighted(sub_img, alpha, im0[rect_y1:rect_y2, rect_x1:rect_x2], 1 - alpha, 0)

        # Put text on the image
        cv2.putText(im0, txt, (text_x, text_y), 0, sf, txt_color, tf, lineType=cv2.LINE_AA)
        text_y_offset = rect_y2

def distance_from_line(A, B, C, x, y):
    return A*x + B*y + C

def calculate_line_coefficients(x1, y1, x2, y2):
    A = y2 - y1
    B = -(x2 - x1)
    C = (x2 - x1) * y1 - (y2 - y1) * x1
    return A, B, C




def process_frame(frame, detector, classifier, tracked_boxes, counts, 
                  x1_point, x2_point, y1_point, y2_point, 
                  tracked_classes, csv_writer, frame_time,
                  heatmap_obj=None):
    results = detector.track(frame, classes=0, conf = 0.25, persist=True, verbose=False, tracker="bytetrack.yaml")
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xywh.cpu().tolist()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        for box, track_id in zip(boxes, track_ids):
            if track_id not in tracked_boxes:
                tracked_boxes[track_id] = {'previous': None, 'current': box}
            else:
                tracked_boxes[track_id]['previous'] = tracked_boxes[track_id]['current']
                tracked_boxes[track_id]['current'] = box

            prev_y = tracked_boxes[track_id]['previous'][1] if tracked_boxes[track_id]['previous'] else None
            prev_x = tracked_boxes[track_id]['previous'][0] if tracked_boxes[track_id]['previous'] else None
            
            curr_y = tracked_boxes[track_id]['current'][1]
            curr_x = tracked_boxes[track_id]['current'][0]  # added for temporary usecase
            
            if prev_y is not None:
                # Define the line coefficients A, B, C (Ax + By + C = 0)
                A, B, C = calculate_line_coefficients(x1_point, y1_point, x2_point, y2_point)

                # Calculate distances from the line for previous and current positions
                prev_distance = distance_from_line(A, B, C, prev_x, prev_y)
                curr_distance = distance_from_line(A, B, C, curr_x, curr_y)

                # Check for crossing the line
                if prev_distance > 0 >= curr_distance and track_id not in counts['down_ids'] and x1_point < curr_x < x2_point:
                    pred_class = classifier_call(box, frame, classifier)
                    counts['down'] += 1
                    counts['down_ids'].append(track_id)
                    age_group = pred_class.split('_')[0]
                    gender = pred_class.split('_')[1]
                    key = f"{gender}, {age_group}"
                    tracked_classes[key] = tracked_classes.get(key, 0) + 1
                    split_class_name = pred_class.split('_')
                    csv_writer.writerow([datetime.fromtimestamp(frame_time).strftime('%Y-%m-%d %H:%M:%S'), split_class_name[0], split_class_name[1]])

                if prev_distance < 0 <= curr_distance and track_id not in counts['up_ids'] and x1_point < curr_x < x2_point:
                    pred_class = classifier_call(box, frame, classifier)
                    counts['up'] += 1
                    counts['up_ids'].append(track_id)
                # if prev_y < horizontal_line_y <= curr_y and track_id not in counts['down_ids'] and x1_point < curr_x < x2_point:
                #     pred_class = classifier_call(box, frame, classifier)
                #     counts['down'] += 1
                #     counts['down_ids'].append(track_id)
                #     age_group = pred_class.split('_')[0]
                #     gender = pred_class.split('_')[1]
                #     key = f"{gender}, {age_group}"
                #     tracked_classes[key] = tracked_classes.get(key, 0) + 1
                #     split_class_name = pred_class.split('_')
                #     csv_writer.writerow([datetime.fromtimestamp(frame_time).strftime('%Y-%m-%d %H:%M:%S'), split_class_name[0],split_class_name[1]])

                # if prev_y > horizontal_line_y >= curr_y and track_id not in counts['up_ids'] and x1_point < curr_x < x2_point:
                #     pred_class = classifier_call(box, frame, classifier)
                #     counts['up'] += 1
                #     counts['up_ids'].append(track_id)
                    # tracked_classes[pred_class] = tracked_classes.get(pred_class, 0) + 1
                    # split_class_name = pred_class.split('_')
                    # csv_writer.writerow([datetime.fromtimestamp(frame_time).strftime('%Y-%m-%d %H:%M:%S'), split_class_name[0], split_class_name[1]])
                    

        frame = results[0].plot(conf = False, labels = True)
        cv2.line(frame, (x1_point, y1_point), (x2_point, y2_point), (255, 255, 0), 5)
            # frame = annotate_frame(frame, track_id, box, counts,x1_point, x2_point, horizontal_line_y)
    
        if heatmap_obj is not None:
            frame = heatmap_obj.generate_heatmap(frame, results)

    return frame