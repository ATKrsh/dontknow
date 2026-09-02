import os
import cv2
import numpy as np
from video_processor import VideoProcessor
from vision_engine import VisionEngine
from summarizer import TextSummarizer

def create_synthetic_test_video(output_video_path: str, fps: float = 5.0):
    """Creates a sample test video with visually distinct scenes."""
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Scene 1: Red Sunset Scene (5 frames)
    for _ in range(5):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :] = (30, 50, 220)
        cv2.circle(frame, (320, 240), 80, (0, 215, 255), -1)
        writer.write(frame)

    # Scene 2: Ocean Blue Scene (5 frames)
    for _ in range(5):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :] = (200, 120, 20)
        cv2.rectangle(frame, (100, 300), (540, 450), (255, 255, 255), -1)
        writer.write(frame)

    writer.release()
    print(f"Created visual test video: {output_video_path}")

def run_tests():
    test_dir = os.path.join(os.path.dirname(__file__), "test_run")
    os.makedirs(test_dir, exist_ok=True)
    video_path = os.path.join(test_dir, "visual_test_video.mp4")
    frames_subfolder = os.path.join(test_dir, "frames_prompt_output")

    print("\n--- 1. Video Frame Extraction ---")
    create_synthetic_test_video(video_path, fps=5.0)
    vp = VideoProcessor(video_path)
    extracted_frames = vp.extract_frames(frames_subfolder, frame_step=1)

    print("\n--- 2. Image-to-Prompt Analysis ---")
    vision = VisionEngine()
    prompts_array = vision.process_frame_list(extracted_frames, detailed_prompts=True)

    print("\n--- 3. Unified Story Summarization (No Timestamps) ---")
    summarizer = TextSummarizer()
    summary = summarizer.generate_story_summary(prompts_array, sentences_count=3)

    print("\nSynthesized Unified Story Output:\n" + summary["story_markdown"])

    assert "Chronological" not in summary["story_markdown"], "Timestamped timeline still present!"
    assert len(summary["summary"]) > 0, "Summary text is empty!"

    print("\n[SUCCESS] UNIFIED STORY SUMMARY TEST PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run_tests()
