import os
import glob
import cv2
import random
import numpy as np

kpz_img_files = glob.glob(r"data\raw\roboflow_kpz2\**\*.jpg", recursive=True)
out_img_dir = r"data\raw\kpz_crop_zoomed\images"
out_lbl_dir = r"data\raw\kpz_crop_zoomed\labels"
os.makedirs(out_img_dir, exist_ok=True)
os.makedirs(out_lbl_dir, exist_ok=True)

random.seed(42)
print(f"Generating Crop-Zoomed samples from {len(kpz_img_files)} CCTV images...")

target_count = 1200
generated = 0

# Shuffle images
random.shuffle(kpz_img_files)

for img_p in kpz_img_files:
    if generated >= target_count:
        break
        
    lbl_p = img_p.replace("images", "labels").replace(".jpg", ".txt")
    if not os.path.exists(lbl_p):
        continue
        
    boxes = []
    with open(lbl_p) as f:
        for line in f:
            p = line.strip().split()
            if len(p) >= 5:
                c = int(p[0])
                xc, yc, w, h = float(p[1]), float(p[2]), float(p[3]), float(p[4])
                # In kpz2: 0 is adult, 1 is child, 2 is elderly
                target_c = 0 if c == 0 else 1
                boxes.append((target_c, xc, yc, w, h))
                
    # We want crops centered around child boxes (target_c == 1)
    child_boxes = [b for b in boxes if b[0] == 1]
    if not child_boxes:
        continue
        
    img = cv2.imread(img_p)
    if img is None:
        continue
    h_img, w_img = img.shape[:2]
    
    # Pick a random child as the zoom target
    chosen_child = random.choice(child_boxes)
    c_xc, c_yc, c_w, c_h = chosen_child[1], chosen_child[2], chosen_child[3], chosen_child[4]
    
    center_x = int(c_xc * w_img)
    center_y = int(c_yc * h_img)
    
    # Crop size: 2.0x to 3.5x the child's height, so the child occupies 30% - 50% of the crop!
    crop_h = int(c_h * h_img * random.uniform(2.0, 3.2))
    crop_w = int(crop_h * random.uniform(0.9, 1.2))
    
    # Ensure crop size is reasonable
    crop_h = max(180, min(crop_h, h_img))
    crop_w = max(180, min(crop_w, w_img))
    
    # Calculate crop coordinates with random jitter around center
    jitter_x = random.randint(-int(crop_w * 0.2), int(crop_w * 0.2))
    jitter_y = random.randint(-int(crop_h * 0.2), int(crop_h * 0.2))
    
    crop_x1 = max(0, min(center_x - crop_w // 2 + jitter_x, w_img - crop_w))
    crop_y1 = max(0, min(center_y - crop_h // 2 + jitter_y, h_img - crop_h))
    crop_x2 = crop_x1 + crop_w
    crop_y2 = crop_y1 + crop_h
    
    cropped_img = img[crop_y1:crop_y2, crop_x1:crop_x2]
    cropped_h, cropped_w = cropped_img.shape[:2]
    
    # Adjust all bounding boxes to the crop coordinates
    valid_crop_boxes = []
    for cls_id, xc, yc, bw, bh in boxes:
        # Convert to absolute pixel coordinates in original image
        bx1 = (xc - bw / 2.0) * w_img
        by1 = (yc - bh / 2.0) * h_img
        bx2 = (xc + bw / 2.0) * w_img
        by2 = (yc + bh / 2.0) * h_img
        
        # Intersect with crop window
        ix1 = max(crop_x1, bx1)
        iy1 = max(crop_y1, by1)
        ix2 = min(crop_x2, bx2)
        iy2 = min(crop_y2, by2)
        
        inter_w = max(0, ix2 - ix1)
        inter_h = max(0, iy2 - iy1)
        orig_area = (bx2 - bx1) * (by2 - by1)
        
        # Only keep box if at least 40% of it is inside the crop
        if orig_area > 0 and (inter_w * inter_h) / orig_area >= 0.40:
            # Convert to normalized coordinates relative to crop
            new_xc = (ix1 + ix2) / 2.0 - crop_x1
            new_yc = (iy1 + iy2) / 2.0 - crop_y1
            new_w = inter_w
            new_h = inter_h
            
            norm_xc = new_xc / float(cropped_w)
            norm_yc = new_yc / float(cropped_h)
            norm_w = new_w / float(cropped_w)
            norm_h = new_h / float(cropped_h)
            
            valid_crop_boxes.append(f"{cls_id} {norm_xc:.6f} {norm_yc:.6f} {norm_w:.6f} {norm_h:.6f}\n")
            
    if not valid_crop_boxes:
        continue
        
    out_fname = f"cz_{generated:04d}_{os.path.basename(img_p)}"
    cv2.imwrite(os.path.join(out_img_dir, out_fname), cropped_img)
    with open(os.path.join(out_lbl_dir, out_fname.replace(".jpg", ".txt")), "w") as f:
        f.writelines(valid_crop_boxes)
        
    generated += 1

print(f"Generated {generated} Crop-Zoomed samples in {out_img_dir}!")
