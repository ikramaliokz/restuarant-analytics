import cv2

input_filename = '/Users/muhammadbaqir/Downloads/MOT20-02-raw.webm'
output_filename = 'test_videos/test3.mp4'

cap = cv2.VideoCapture(input_filename)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

out = cv2.VideoWriter(output_filename, fourcc, fps, frame_size)

while True:
    ret, frame = cap.read()

    if ret:
        out.write(frame)
    else:
        break

cap.release()
out.release()

print(f'Video conversion complete. File saved as {output_filename}')
