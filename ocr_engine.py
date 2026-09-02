import os
import cv2
import logging
from typing import List, Dict, Any, Optional, Callable

logger = logging.getLogger("dontknow.ocr")

class OCREngine:
    """
    Local Image-to-Text (OCR) Engine powered by RapidOCR (ONNX Runtime).
    Processes frame PNGs sequentially from frame 1 to the end.
    """
    def __init__(self):
        self._engine = None
        self._init_engine()

    def _init_engine(self):
        try:
            from rapidocr_onnxruntime import RapidOCR
            self._engine = RapidOCR()
            logger.info("RapidOCR ONNX engine initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize RapidOCR engine: {e}")
            self._engine = None

    def extract_text_from_image(self, image_path: str, min_confidence: float = 0.4) -> Dict[str, Any]:
        """
        Run OCR on a single PNG image file.
        
        :param image_path: Path to PNG image file.
        :param min_confidence: Minimum score to accept text line.
        :return: Dict with extracted_text (string), text_lines (list of dicts with box, text, score).
        """
        if not os.path.exists(image_path):
            return {"extracted_text": "", "lines": [], "error": f"File not found: {image_path}"}

        if self._engine is None:
            self._init_engine()

        if self._engine is None:
            return {"extracted_text": "", "lines": [], "error": "OCR Engine unavailable."}

        try:
            result, elapse = self._engine(image_path)
            lines = []
            text_blocks = []

            if result:
                for item in result:
                    # item format: [box, text, score]
                    if len(item) >= 3:
                        box, text, score = item[0], item[1], float(item[2])
                        text_str = str(text).strip()
                        if score >= min_confidence and text_str:
                            lines.append({
                                "box": box,
                                "text": text_str,
                                "score": score
                            })
                            text_blocks.append(text_str)

            combined_text = " ".join(text_blocks)
            return {
                "extracted_text": combined_text,
                "lines": lines,
                "elapse": elapse if isinstance(elapse, (list, tuple, float, int)) else 0.0
            }

        except Exception as e:
            logger.exception(f"OCR Error on {image_path}: {e}")
            return {"extracted_text": "", "lines": [], "error": str(e)}

    def process_frame_list(
        self,
        frames_info: List[Dict[str, Any]],
        min_confidence: float = 0.4,
        progress_callback: Optional[Callable[[int, int, Dict[str, Any]], None]] = None,
        stop_checker: Optional[Callable[[], bool]] = None
    ) -> List[Dict[str, Any]]:
        """
        Processes all PNG frame images in sequential order from first to last frame.
        Stores all extracted text from all frames in an array.
        
        :param frames_info: List of dicts containing image_path, frame_index, timestamp_sec, etc.
        :param min_confidence: Minimum score for text lines.
        :param progress_callback: Callable receiving (current_idx, total_count, frame_text_record).
        :param stop_checker: Callable checking if task should cancel.
        :return: Array of text records for all frames.
        """
        text_array = []
        total = len(frames_info)

        for idx, frame in enumerate(frames_info, start=1):
            if stop_checker and stop_checker():
                break

            img_path = frame.get("image_path", "")
            ocr_result = self.extract_text_from_image(img_path, min_confidence=min_confidence)

            record = {
                "frame_index": frame.get("frame_index", idx),
                "timestamp_sec": frame.get("timestamp_sec", 0.0),
                "filename": frame.get("filename", os.path.basename(img_path)),
                "image_path": img_path,
                "text": ocr_result["extracted_text"],
                "lines_count": len(ocr_result.get("lines", [])),
                "lines": ocr_result.get("lines", [])
            }

            text_array.append(record)

            if progress_callback:
                progress_callback(idx, total, record)

        return text_array
