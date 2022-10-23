import os
import numpy as np
import shutil
import tqdm


for class_name in tqdm.tqdm(os.listdir('videos/pre-processed/train'), desc='Iterate classes'):
    videos = os.listdir(os.path.join('videos/pre-processed/train', class_name))
    val_nums = int(len(videos) * 0.1)
    if val_nums <= 0:
        tqdm.tqdm.write(f'Insufficient videos: {class_name}, {len(videos)}')
        continue
    val_videos = np.random.choice(videos, int(len(videos) * 0.1), replace=False)
    tqdm.tqdm.write(f'{class_name}: {len(val_videos)}')
    for val_video in tqdm.tqdm(val_videos, desc='Iterate videos', leave=False):
        target_dir = os.path.join('videos/pre-processed/val', class_name)
        os.makedirs(target_dir, exist_ok=True)
        shutil.move(os.path.join('videos/pre-processed/train', class_name, val_video), os.path.join(target_dir, val_video))

