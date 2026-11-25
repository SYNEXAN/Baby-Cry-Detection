import os
import glob
import cv2
from ultralytics import YOLO

frames_dir = r"data\raw\classroom_video_frames"
labels_dir = r"data\raw\classroom_video_labels"
vis_dir = r"data\raw\classroom_video_vis"
os.makedirs(labels_dir, exist_ok=True)
os.makedirs(vis_dir, exist_ok=True)

pose_model = YOLO("yolov8n-pose.pt")
frame_files = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))

print(f"Generating ground-truth annotations for {len(frame_files)} classroom frames...")

total_adults = 0
total_children = 0

for img_p in frame_files:
    fname = os.path.basename(img_p)
    img = cv2.imread(img_p)
    h_img, w_img = img.shape[:2]
    
    res = pose_model(img, conf=0.3, iou=0.5, verbose=False)[0]
    
    yolo_lines = []
    vis_img = img.copy()
    
    for b in res.boxes:
        x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
        box_w = x2 - x1
        box_h = y2 - y1
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        
        # Determine if this box is the teacher (adult woman):
        # Characteristics of the teacher in this video:
        # 1. Standing upright in the upper/background half (y1 < 100) with height > 160
        # 2. Or walking in the background near tables with height > 220
        # 3. Y-position: her feet/torso reach down while head is near top (y1 < 120, y2 > 200, box_h > 180)
        # All other people in this classroom are children (sitting at desks, drawing, or small boy walking)
        
        is_teacher = False
        if y1 < 90 and box_h > 165 and y2 < 380:
            is_teacher = True
        elif y1 < 120 and box_h > 210:
            is_teacher = True
            
        cls_id = 0 if is_teacher else 1
        cls_name = "adult" if cls_id == 0 else "baby_child"
        color = (0, 200, 0) if cls_id == 0 else (0, 140, 255) # Green adult, Orange child
        
        if cls_id == 0:
            total_adults += 1
        else:
            total_children += 1
            
        # Normalized YOLO format: class xc yc w h
        xc = (x1 + x2) / 2.0 / w_img
        yc = (y1 + y2) / 2.0 / h_img
        norm_w = box_w / float(w_img)
        norm_h = box_h / float(h_img)
        
        yolo_lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {norm_w:.6f} {norm_h:.6f}\n")
        
        # Draw on vis
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis_img, cls_name, (x1+3, max(20, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
    # Write label file
    lbl_p = os.path.join(labels_dir, fname.replace(".jpg", ".txt"))
    with open(lbl_p, "w") as f:
        f.writelines(yolo_lines)
        
    # Save visual check
    cv2.imwrite(os.path.join(vis_dir, fname), vis_img)

print(f"Annotation complete! Total frames: {len(frame_files)}")
print(f"Total annotations -> Adult: {total_adults} | Baby_Child: {total_children}")
print(f"Labels saved to: {labels_dir}")
print(f"Visual debug checks saved to: {vis_dir}")
