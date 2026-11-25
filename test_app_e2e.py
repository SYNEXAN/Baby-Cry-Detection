"""
E2E Verification Script for app.py logic and components
"""
import os
import cv2
from app import process_single_image, stream_video_feed, SAMPLE_IMAGES, SAMPLE_VIDEOS

def test_image_pipeline():
    print("🧪 [Test 1] Testing Single Image Inference (Baby Alone)...")
    img_bgr = cv2.imread(SAMPLE_IMAGES["baby_alone"])
    assert img_bgr is not None, "Failed to load baby alone sample"
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    annotated_img, status_html, stat_cards, df = process_single_image(img_rgb, conf_thresh=0.35)
    assert annotated_img is not None, "Output image is None"
    assert "danger" in status_html.lower() or "cảnh báo" in status_html.lower() or "nguy hiểm" in status_html.lower(), f"Expected danger banner, got: {status_html}"
    print("✅ Baby Alone image correctly flagged as DANGER / UNATTENDED!")
    print("Dataframe result:\n", df)

    print("\n🧪 [Test 2] Testing Single Image Inference (Child + Adult)...")
    img2_bgr = cv2.imread(SAMPLE_IMAGES["child_and_adult"])
    img2_rgb = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2RGB)
    annotated_img2, status_html2, stat_cards2, df2 = process_single_image(img2_rgb, conf_thresh=0.35)
    assert annotated_img2 is not None
    assert "an toàn" in status_html2.lower() or "safe" in status_html2.lower() or "người lớn" in status_html2.lower(), f"Expected SAFE, got: {status_html2}"
    print("✅ Child + Adult image correctly flagged as SAFE!")
    print("Dataframe result:\n", df2)

def test_video_pipeline():
    print("\n🧪 [Test 3] Testing Video Stream Processing Pipeline...")
    vid_path = SAMPLE_VIDEOS["baby_alone"]
    generator = stream_video_feed(
        video_input=vid_path,
        file_video_input=None,
        conf_thresh=0.35,
        frame_skip=2,
        max_duration=3,
        tracker_name="ByteTrack",
        enable_trail=True
    )
    
    frame_count = 0
    last_status = ""
    for frame_out, status_html, stat_cards, df, audit_md in generator:
        if frame_out is not None:
            frame_count += 1
            last_status = status_html
            
    assert frame_count > 0, "No frames were processed by stream generator"
    print(f"✅ Video stream successfully processed {frame_count} frames!")
    print(f"Final Status: {last_status[:100]}...")

if __name__ == "__main__":
    test_image_pipeline()
    test_video_pipeline()
    print("\n🎉 ALL E2E TESTS PASSED SUCCESSFULLY!")
