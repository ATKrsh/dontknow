import os
import cv2
import time
import numpy as np
from typing import Callable, Optional, Dict, Any, List

class VideoProcessor:
    """
    Handles video opening, metadata extraction, and frame-by-frame export to PNG files.
    Fully compatible with Windows paths, non-ASCII characters, and spaces.
    """
    def __init__(self, video_path: str):
        self.video_path = os.path.abspath(os.path.normpath(video_path))
        if not os.path.exists(self.video_path):
            raise FileNotFoundError(f"Video file not found: {self.video_path}")

    @staticmethod
    def safe_imwrite(path: str, frame) -> bool:
        """Robust image saving for Windows handling non-ASCII/spaces in paths."""
        try:
            is_success, buffer = cv2.imencode(".png", frame)
            if is_success:
                with open(path, "wb") as f:
                    f.write(buffer)
                return True
        except Exception:
            pass
        return bool(cv2.imwrite(path, frame))

    def get_metadata(self) -> Dict[str, Any]:
        """Extract video metadata using OpenCV."""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video file: {self.video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
        duration_sec = total_frames / fps if fps > 0 else 0.0

        cap.release()
        return {
            "path": self.video_path,
            "filename": os.path.basename(self.video_path),
            "fps": fps,
            "total_frames": total_frames,
            "width": width,
            "height": height,
            "duration_sec": duration_sec
        }

    def extract_frames(
        self,
        output_dir: str,
        frame_step: int = 1,
        progress_callback: Optional[Callable[[int, int, str, float], None]] = None,
        stop_checker: Optional[Callable[[], bool]] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract frames from frame 1 to the end, saving each as PNG in output_dir.
        
        :param output_dir: Subfolder directory to save PNGs.
        :param frame_step: 1 = every frame, 2 = every 2nd frame, etc.
        :param progress_callback: Callable receiving (frame_index, total_frames, saved_png_path, timestamp_sec).
        :param stop_checker: Callable returning True if extraction should be cancelled.
        :return: List of frame info dicts saved.
        """
        abs_output_dir = os.path.abspath(os.path.normpath(output_dir))
        os.makedirs(abs_output_dir, exist_ok=True)

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video for frame extraction: {self.video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

        extracted_frames = []
        current_frame_idx = 0

        while True:
            if stop_checker and stop_checker():
                break

            ret, frame = cap.read()
            if not ret:
                break

            current_frame_idx += 1  # 1-indexed frame counting

            if (current_frame_idx - 1) % frame_step == 0:
                timestamp_sec = round((current_frame_idx - 1) / fps, 3)
                filename = f"frame_{current_frame_idx:06d}.png"
                frame_path = os.path.join(abs_output_dir, filename)

                # Save frame as PNG safely
                saved_ok = self.safe_imwrite(frame_path, frame)
                if not saved_ok or not os.path.exists(frame_path):
                    # Direct cv2 fallback
                    cv2.imwrite(frame_path, frame)

                info = {
                    "frame_index": current_frame_idx,
                    "timestamp_sec": timestamp_sec,
                    "image_path": frame_path,
                    "filename": filename
                }
                extracted_frames.append(info)

                if progress_callback:
                    progress_callback(current_frame_idx, total_frames, frame_path, timestamp_sec)

        cap.release()
        return extracted_frames
