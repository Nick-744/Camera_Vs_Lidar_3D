import os
import shutil

def main():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    all_directories = [d for d in os.listdir(current_directory)
                       if os.path.isdir(os.path.join(current_directory, d))]
    all_directories.sort()

    rgbs_directories  = []
    masks_directories = []
    for directory in all_directories:
        mask_dir = os.path.join(current_directory, directory, 'masks')
        if os.path.exists(mask_dir):
            masks_directories.append(mask_dir)
        
        rgb_dir = os.path.join(current_directory, directory, 'rgb')
        if os.path.exists(rgb_dir):
            rgbs_directories.append(rgb_dir)
    rgbs_directories.sort()
    masks_directories.sort()

    output_rgb_dir  = os.path.join(current_directory, 'merged', 'rgb')
    output_mask_dir = os.path.join(current_directory, 'merged', 'masks')
    os.makedirs(output_rgb_dir, exist_ok = True)
    os.makedirs(output_mask_dir, exist_ok = True)

    counter = 0
    for (dir_index, (rgb_dir, mask_dir)) in enumerate(zip(rgbs_directories, masks_directories)):
        rgb_files  = sorted(os.listdir(rgb_dir))
        mask_files = sorted(os.listdir(mask_dir))

        if len(rgb_files) != len(mask_files):
            print(f'Warning: {rgb_dir} and {mask_dir} have different number of files!')
            continue;

        for (file_index, (rgb_file, mask_file)) in enumerate(zip(rgb_files, mask_files)):
            new_name = f'{counter:05d}.png'
            counter += 1

            src_rgb_path  = os.path.join(rgb_dir, rgb_file)
            src_mask_path = os.path.join(mask_dir, mask_file)

            dst_rgb_path  = os.path.join(output_rgb_dir, new_name)
            dst_mask_path = os.path.join(output_mask_dir, new_name)

            shutil.copyfile(src_rgb_path, dst_rgb_path)
            shutil.copyfile(src_mask_path, dst_mask_path)

        print(f'Processed {len(rgb_files) + len(mask_files)} files from {rgb_dir} and {mask_dir}')

    return;

if __name__ == "__main__":
    main()
