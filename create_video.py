import cv2

base_dir = '/Users/muhammadbaqir/Downloads/MOT20Det/train/MOT20-01'

seq_info_path = f'{base_dir}/seqinfo.ini'

# Read the sequence information
with open(seq_info_path, 'r') as file:
    lines = file.readlines()
    seq_info = {}
    for line in lines:
        # skip first line
        if line.startswith('['):
            continue
        print(line)
        key, value = line.strip().split('=')
        seq_info[key] = value

        if key == 'imExt':
            break

output_filename = f'test_videos/test_{seq_info['name']}.mp4'

out_writer = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*'mp4v'), int(seq_info['frameRate']), 
                             (int(seq_info['imWidth']), int(seq_info['imHeight'])))


for i in range(int(seq_info['seqLength'])):
    img_path = f"{base_dir}/{seq_info['imDir']}/{str(i+1).zfill(6)}{seq_info['imExt']}"
    img = cv2.imread(img_path)
    out_writer.write(img)

out_writer.release()
print(f'Video conversion complete. File saved as {output_filename}')
