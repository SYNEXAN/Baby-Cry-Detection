import os
import time
import cv2
from pathlib import Path
from ultralytics import YOLO

MODEL_PATH = r"kaggle_results\v4_clean_balanced\best.pt"
OUT_DIR = Path(r"output_videos\v4_clean_balanced")
OUT_DIR.mkdir(parents=True, exist_ok=True)

VIDEOS = [
    ("v4_sample_baby_alone.mp4", r"demo_videos\sample_baby_alone.mp4"),
]
for p in sorted(Path("VIDEO_TEST").glob("*.mp4")):
    VIDEOS.append((f"v4_{p.name}", str(p)))

print(f"Loading model: {MODEL_PATH}")
model = YOLO(MODEL_PATH)
print("Model classes:", model.names)

for out_name, in_path in VIDEOS:
    if not os.path.exists(in_path):
        print(f"SKIP missing: {in_path}")
        continue

    cap = cv2.VideoCapture(in_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Dynamic imgsz: 640 for low-res <=640px videos, 960 for HD/960p/4K videos
    dyn_imgsz = 640 if max(w, h) <= 640 else 960
    dyn_conf = 0.30 if max(w, h) <= 640 else 0.40

    out_w, out_h = w, h
    if w > 1280:
        scale = 1280 / w
        out_w = 1280
        out_h = int(h * scale)
        out_h = out_h - (out_h % 2)

    out_path = str(OUT_DIR / out_name)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (out_w, out_h))

    t0 = time.time()
    frame_idx = 0
    total_adults = 0
    total_kids = 0
    frames_with_adults = 0
    frames_with_kids = 0

    print(f"\n>>> Rendering [{out_name}] ({total} frames, {w}x{h} -> {out_w}x{out_h}, imgsz={dyn_imgsz}, conf={dyn_conf})...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if out_w != w:
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

        res = model.predict(
            frame,
            imgsz=dyn_imgsz,
            conf=dyn_conf,
            iou=0.45,
            agnostic_nms=True,
            verbose=False,
        )[0]

        n_a, n_k = 0, 0
        for box in res.boxes:
            cls = int(box.cls[0])
            cf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if cls == 0:
                n_a += 1
                color = (235, 130, 30)
                label = f"adults {cf:.2f}"
            else:
                n_k += 1
                color = (0, 215, 255)
                label = f"kids {cf:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            ty = max(th + 6, y1)
            cv2.rectangle(frame, (x1, ty - th - 6), (x1 + tw + 6, ty), color, -1)
            cv2.putText(frame, label, (x1 + 3, ty - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

        total_adults += n_a
        total_kids += n_k
        if n_a > 0:
            frames_with_adults += 1
        if n_k > 0:
            frames_with_kids += 1

        hud = f"YOLO11-M V4 ({dyn_imgsz}) | Frame {frame_idx+1}/{total} | Adults: {n_a} | Kids: {n_k}"
        cv2.rectangle(frame, (8, 8), (540, 36), (20, 20, 20), -1)
        cv2.putText(frame, hud, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    dt = time.time() - t0
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(
        f"DONE [{out_name}] in {dt:.1f}s ({size_mb:.2f} MB) | "
        f"Adults in {frames_with_adults}/{frame_idx} frames (avg {total_adults/max(1,frame_idx):.2f}/f) | "
        f"Kids in {frames_with_kids}/{frame_idx} frames (avg {total_kids/max(1,frame_idx):.2f}/f)"
    )

print("\nALL V4 INFERENCE VIDEOS COMPLETED!")
