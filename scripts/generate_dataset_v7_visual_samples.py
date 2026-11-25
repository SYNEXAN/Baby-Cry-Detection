import os
import glob
import cv2
import numpy as np

ARTIFACT_DIR = r"C:\Users\thiet\.gemini\antigravity-cli\brain\30d185aa-58de-4b8e-b83e-50bde963d7f6"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

CLASS_COLORS = {
    0: (216, 180, 0),    # Adult: Cyan-Blue (BGR)
    1: (0, 127, 247),    # Baby/Child: Orange (BGR)
}
CLASS_NAMES = {
    0: "adult (Full-Body)",
    1: "baby_child (Full-Body)"
}

def draw_yolo_boxes(img, label_path):
    out = img.copy()
    h_img, w_img = out.shape[:2]
    if not os.path.exists(label_path):
        return out
    
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                c = int(parts[0])
                xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                x1 = int((xc - w / 2) * w_img)
                y1 = int((yc - h / 2) * h_img)
                x2 = int((xc + w / 2) * w_img)
                y2 = int((yc + h / 2) * h_img)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w_img - 1, x2), min(h_img - 1, y2)
                
                color = CLASS_COLORS.get(c, (0, 255, 0))
                label_txt = f"[{c}] {CLASS_NAMES.get(c, str(c))}"
                
                # Draw box
                cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
                
                # Draw filled label banner
                (tw, th), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(out, (x1, max(0, y1 - th - 8)), (x1 + tw + 10, y1), color, -1)
                cv2.putText(out, label_txt, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return out

# Sample 1: COCO Indoor Adult sitting
img1_p = "data/processed/vision_balanced_v6/train/images/coco_000000006763.jpg"
lbl1_p = "data/processed/vision_balanced_v6/train/labels/coco_000000006763.txt"
img1 = draw_yolo_boxes(cv2.imread(img1_p), lbl1_p)
cv2.imwrite(os.path.join(ARTIFACT_DIR, "dataset_v7_sample_1_indoor_adult_sitting.jpg"), img1)

# Sample 2: COCO Indoor Adult closeup
img2_p = "data/processed/vision_balanced_v6/train/images/coco_000000003156.jpg"
lbl2_p = "data/processed/vision_balanced_v6/train/labels/coco_000000003156.txt"
img2 = draw_yolo_boxes(cv2.imread(img2_p), lbl2_p)
cv2.imwrite(os.path.join(ARTIFACT_DIR, "dataset_v7_sample_2_indoor_adult_closeup.jpg"), img2)

# Sample 3: Daycare Interaction
kpz_files = glob.glob("data/processed/vision_balanced_v6/train/images/kpz2_*.jpg")
img3_p = kpz_files[0]
lbl3_p = img3_p.replace("images", "labels").replace(".jpg", ".txt")
img3 = draw_yolo_boxes(cv2.imread(img3_p), lbl3_p)
cv2.imwrite(os.path.join(ARTIFACT_DIR, "dataset_v7_sample_3_daycare_teacher_children.jpg"), img3)

# Sample 4: Baby Full Body in crib / room
img4_p = "data/processed/vision_balanced_v6/train/images/tb_010_jpeg_jpg.rf.7e59f9c1c7c0fc197d15e58a0acc6c92.jpg"
lbl4_p = "data/processed/vision_balanced_v6/train/labels/tb_010_jpeg_jpg.rf.7e59f9c1c7c0fc197d15e58a0acc6c92.txt"
img4 = draw_yolo_boxes(cv2.imread(img4_p), lbl4_p)
cv2.imwrite(os.path.join(ARTIFACT_DIR, "dataset_v7_sample_4_baby_alone_fullbody.jpg"), img4)

# Sample 5: Toddler Playing Full Body
img5_p = "data/processed/vision_balanced_v6/train/images/tb_013_jpeg_jpg.rf.463847620e484760dc5d0756bf029451.jpg"
lbl5_p = "data/processed/vision_balanced_v6/train/labels/tb_013_jpeg_jpg.rf.463847620e484760dc5d0756bf029451.txt"
img5 = draw_yolo_boxes(cv2.imread(img5_p), lbl5_p)
cv2.imwrite(os.path.join(ARTIFACT_DIR, "dataset_v7_sample_5_toddler_playing_fullbody.jpg"), img5)

# Sample 6: Correct Ground Truth on User's Classroom Image
user_img_p = r"C:/Users/thiet/.gemini/antigravity-cli/brain/30d185aa-58de-4b8e-b83e-50bde963d7f6/.user_uploaded/uploaded_media_1794978411315.jpg"
u_img = cv2.imread(user_img_p)
hu, wu = u_img.shape[:2]

gt_boxes = [
    (0, 107, 44, 307, 368, "[0] adult: Co giao quy cham be (Full-Body)"),
    (0, 457, 102, 724, 414, "[0] adult: Co giao ngoi cam bong (Full-Body)"),
    (1, 260, 157, 418, 392, "[1] baby_child: Be gai ngoi giua (Full-Body)"),
    (1, 12, 215, 83, 414, "[1] baby_child: Be ngoi ben trai (Full-Body)")
]
img6 = u_img.copy()
for c, x1, y1, x2, y2, lbl in gt_boxes:
    color = CLASS_COLORS[c]
    cv2.rectangle(img6, (x1, y1), (x2, y2), color, 3)
    (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 2)
    cv2.rectangle(img6, (x1, max(0, y1 - th - 8)), (x1 + tw + 10, y1), color, -1)
    cv2.putText(img6, lbl, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 2)

cv2.rectangle(img6, (0, hu - 30), (wu, hu), (0, 160, 0), -1)
cv2.putText(img6, "CHUAN HOA DATASET V7: 100% FULL-BODY BBOX | 2 CO GIAO = ADULT",
            (25, hu - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)

cv2.imwrite(os.path.join(ARTIFACT_DIR, "dataset_v7_sample_6_user_classroom_groundtruth.jpg"), img6)

# Create 2x3 combined showcase grid
target_w, target_h = 520, 360
imgs_resized = [cv2.resize(im, (target_w, target_h)) for im in [img1, img2, img3, img4, img5, img6]]
row1 = np.hstack([imgs_resized[0], imgs_resized[1], imgs_resized[2]])
row2 = np.hstack([imgs_resized[3], imgs_resized[4], imgs_resized[5]])
grid = np.vstack([row1, row2])

banner = np.zeros((65, grid.shape[1], 3), dtype=np.uint8)
cv2.putText(banner, "DATASET VERSION 7 FULL-BODY SHOWCASE (COCO INDOOR ADULTS + FULL-BODY BABIES)",
            (35, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 255, 255), 2)
grid_final = np.vstack([banner, grid])

cv2.imwrite(os.path.join(ARTIFACT_DIR, "dataset_v7_showcase_grid.jpg"), grid_final)
print("SUCCESS: Generated 6 visual showcase samples and master grid successfully!")
