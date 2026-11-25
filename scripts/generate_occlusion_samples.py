import os
import cv2
import numpy as np

ARTIFACT_DIR = r"C:\Users\thiet\.gemini\antigravity-cli\brain\30d185aa-58de-4b8e-b83e-50bde963d7f6"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# Sample 7: Toddler in play area with Random Erasing / Occlusion patch
img_p = "data/processed/vision_balanced_v6/train/images/tb_013_jpeg_jpg.rf.463847620e484760dc5d0756bf029451.jpg"
lbl_p = "data/processed/vision_balanced_v6/train/labels/tb_013_jpeg_jpg.rf.463847620e484760dc5d0756bf029451.txt"

img7 = cv2.imread(img_p)
h7, w7 = img7.shape[:2]

# Apply simulated occlusion / Cutout (e.g. crib bar or furniture block)
occ7 = img7.copy()
# Draw simulated crib bars or furniture occlusion
cv2.rectangle(occ7, (int(w7 * 0.35), int(h7 * 0.4)), (int(w7 * 0.65), int(h7 * 0.75)), (115, 115, 115), -1)
cv2.putText(occ7, "OCCLUSION MASK (Cutout / Erasing)", (int(w7 * 0.1), int(h7 * 0.35)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 2)

# Draw ground truth bbox encompassing the whole toddler
with open(lbl_p) as f:
    for line in f:
        p = line.strip().split()
        if len(p) >= 5:
            xc, yc, w, h = float(p[1]), float(p[2]), float(p[3]), float(p[4])
            x1 = int((xc - w / 2) * w7)
            y1 = int((yc - h / 2) * h7)
            x2 = int((xc + w / 2) * w7)
            y2 = int((yc + h / 2) * h7)
            cv2.rectangle(occ7, (x1, y1), (x2, y2), (0, 127, 247), 3)
            lbl_txt = "[1] baby_child (Bi che khuat 40%)"
            (tw, th), _ = cv2.getTextSize(lbl_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 2)
            cv2.rectangle(occ7, (x1, max(0, y1 - th - 6)), (x1 + tw + 8, y1), (0, 127, 247), -1)
            cv2.putText(occ7, lbl_txt, (x1 + 4, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)

cv2.imwrite(os.path.join(ARTIFACT_DIR, "dataset_v7_sample_7_child_occlusion_cutout.jpg"), occ7)

# Sample 8: User Classroom Image with Realistic Balloon & Child Overlap Occlusion
user_img_p = r"C:/Users/thiet/.gemini/antigravity-cli/brain/30d185aa-58de-4b8e-b83e-50bde963d7f6/.user_uploaded/uploaded_media_1794978411315.jpg"
img8 = cv2.imread(user_img_p)
h8, w8 = img8.shape[:2]

# Annotated ground truth illustrating handling of severe natural occlusion (Pink Balloon occluding teacher and baby behind)
occ8 = img8.copy()

# Teacher 1 (left) kneeling - green bbox full body
cv2.rectangle(occ8, (107, 44), (307, 368), (0, 216, 180), 3)
cv2.putText(occ8, "[0] adult (Co giao quy - bi be che khuat phan dui)", (107, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 2)

# Teacher 2 (right) - occluded by large pink balloon
cv2.rectangle(occ8, (457, 102), (724, 414), (0, 216, 180), 3)
cv2.putText(occ8, "[0] adult (Co giao bi qua bong che khuat phan than)", (430, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 2)

# Child 1 (center)
cv2.rectangle(occ8, (260, 157), (418, 392), (0, 127, 247), 3)
cv2.putText(occ8, "[1] baby_child (Be gai)", (260, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 127, 247), 2)

# Child 2 behind balloon (heavy occlusion)
cv2.rectangle(occ8, (390, 200), (510, 414), (0, 127, 247), 3)
cv2.putText(occ8, "[1] baby_child (Be sau qua bong)", (380, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 127, 247), 2)

# Banner at top
banner = np.zeros((35, w8, 3), dtype=np.uint8)
cv2.putText(banner, "XU LY CHE KHUAT (NATURAL & SYNTHETIC OCCLUSION HANDLING)", (25, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
occ8_final = np.vstack([banner, occ8])

cv2.imwrite(os.path.join(ARTIFACT_DIR, "dataset_v7_sample_8_classroom_balloon_occlusion.jpg"), occ8_final)
print("SUCCESS: Generated occlusion samples 7 and 8 successfully!")
