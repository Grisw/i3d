import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--beedog_path', type=str)
    parser.add_argument('-d', '--dataset', type=str, default='mpii_cooking')

    args = parser.parse_args()

    beedog_clips_path = os.path.join(args.beedog_path, 'data', args.dataset, 'pre_processed', 'video_clip')
    if not os.path.exists(beedog_clips_path):
        raise FileNotFoundError(f'beedog clips path not found: {beedog_clips_path}')

    class_names = set()
    for video in os.listdir(beedog_clips_path):
        video_path = os.path.join(beedog_clips_path, video)
        for clip in os.listdir(video_path):
            class_name = str(clip).split('.')[0].split('_')[0]
            class_names.add(class_name)

            dst_dir = os.path.join('videos', 'raw', class_name)
            os.makedirs(dst_dir, exist_ok=True)

            os.symlink(os.path.join(video_path, clip), os.path.join(dst_dir, f'{video}_{clip}'))

    with open('classes.txt', 'w') as f:
        for class_name in class_names:
            f.write(f'{class_name}\n')

    print(class_names)


if __name__ == '__main__':
    main()
