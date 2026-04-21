import os
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm
from dataset import get_deterministic_splits

def preprocess_and_save_images(root_dir, split_pieces, save_dir='./local_data'):
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving preprocessed images to: {save_dir}")

    TARGET_W = 416
    TARGET_H = 128
    processed_count = 0

    print(f"Processing {len(split_pieces)} pieces...")

    for piece_name in tqdm(split_pieces, desc="Extracting Staff Lines"):
        piece_dir = os.path.join(root_dir, piece_name)
        if not os.path.isdir(piece_dir): continue

        score_dir_base = os.path.join(piece_dir, 'scores')
        score_folders = os.listdir(score_dir_base)
        if not score_folders: continue
        score_path = os.path.join(score_dir_base, score_folders[0])

        img_dir = os.path.join(score_path, 'img')
        coords_dir = os.path.join(score_path, 'coords')

        if not os.path.exists(img_dir) or not os.path.exists(coords_dir): continue

        for img_file in glob.glob(os.path.join(img_dir, '*.png')):
            page_id = os.path.basename(img_file).replace('.png', '')
            systems_npy_path = os.path.join(coords_dir, f"systems_{page_id}.npy")

            if not os.path.exists(systems_npy_path): continue

            system_boxes = np.load(systems_npy_path)

            with Image.open(img_file) as img:
                img = img.convert('L')
                img_width, img_height = img.size

                for box in system_boxes:
                    if np.ptp(box[:, 0]) < np.ptp(box[:, 1]):
                        sys_top, sys_bottom = float(np.min(box[:, 0])), float(np.max(box[:, 0]))
                    else:
                        sys_top, sys_bottom = float(np.min(box[:, 1])), float(np.max(box[:, 1]))

                    crop_top = max(0, int(sys_top - 30))
                    crop_bottom = min(img_height, int(sys_bottom + 30))

                    crop_img = img.crop((0, crop_top, img_width, crop_bottom))
                    orig_w, orig_h = crop_img.size

                    scale = min(TARGET_W / orig_w, TARGET_H / orig_h)
                    new_w = int(orig_w * scale)
                    new_h = int(orig_h * scale)

                    resized_img = crop_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

                    canvas = Image.new('L', (TARGET_W, TARGET_H), color=255)

                    paste_x = (TARGET_W - new_w) // 2
                    paste_y = (TARGET_H - new_h) // 2
                    canvas.paste(resized_img, (paste_x, paste_y))

                    line_id = f"{piece_name}_{page_id}_{sys_top}"
                    save_path = os.path.join(save_dir, f"{line_id}.png")

                    canvas.save(save_path, format="PNG", optimize=True)
                    processed_count += 1

    print(f"\nSuccessfully generated and saved {processed_count} images to {save_dir}!")

if __name__ == "__main__":
    DATASET_ROOT = './msmd_dataset/msmd_aug_v1-1_no-audio'
    LOCAL_SAVE_DIR = './local_data'
    
    # get  same splits used in training 
    train_pieces, val_pieces, test_pieces = get_deterministic_splits(DATASET_ROOT)
    
    # Process all splits into the local directory
    preprocess_and_save_images(DATASET_ROOT, train_pieces, save_dir=LOCAL_SAVE_DIR)
    preprocess_and_save_images(DATASET_ROOT, val_pieces, save_dir=LOCAL_SAVE_DIR)
    preprocess_and_save_images(DATASET_ROOT, test_pieces, save_dir=LOCAL_SAVE_DIR)