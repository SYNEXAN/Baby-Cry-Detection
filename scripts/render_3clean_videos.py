import os
import time
import cv2
from pathlib import Path
from ultralytics import YOLO

MODEL_PATH = r"kaggle_results\3clean_kids_adults\best.pt"
OUT_DIR = Path(r"output_videos\3clean_kids_adults")
OUT_DIR.mkdir(parents=True, exist_ok=True)

VIDEOS = [
    ("3clean_sample_baby_alone.mp4", r"demo_videos\sample_baby_alone.mp4"),
]
for p in sorted(Path("VIDEO_TEST").glob("*.mp4")):
    VIDEOS.append((f"3clean_{p.name}", str(p)))

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

    # Downscale 4K to 1280x720 for smooth playback & fast encoding while preserving sharpness
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

    print(f"\n>>> Rendering [{out_name}] ({total} frames, {w}x{h} -> {out_w}x{out_h} @ {fps:.1f} fps)...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if out_w != w:
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

        res = model.predict(
            frame,
            imgsz=768,
            conf=0.40,
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
                color = (235, 130, 30)  # Blue-ish BGR for adults
                label = f"adults {cf:.2f}"
            else:
                n_k += 1
                color = (0, 215, 255)   # Gold/Yellow BGR for kids
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

        # Compact HUD bar
        hud = f"YOLO11-M 3-Clean | Frame {frame_idx+1}/{total} | Adults: {n_a} | Kids: {n_k}"
        cv2.rectangle(frame, (8, 8), (520, 36), (20, 20, 20), -1)
        cv2.putText(frame, hud, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

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

print("\nALL 3-CLEAN INFERENCE VIDEOS COMPLETED!")
