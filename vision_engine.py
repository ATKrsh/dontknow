import os
import re
import cv2
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Callable
from PIL import Image

logger = logging.getLogger("dontknow.vision")


def clean_to_the_point_text(text: str) -> str:
    """Removes unused tokens and conversational filler prefixes to make captions clean and direct."""
    if not text:
        return ""
    
    # 1. Remove tokenizer unused token artifacts like '[ unused0 ]'
    cleaned = re.sub(r"\[\s*unused\d+\s*\]", "", text, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r",\s*,", ",", cleaned).strip()
    
    # 2. Remove filler prefixes
    filler_patterns = [
        r"^(this\s+is\s+)?(an?\s+)?(detailed\s+)?(photograph|photo|image|picture|graphic|illustration)\s+(showing|of|depicting|with)?\s*",
        r"^(an?\s+)?(photograph|photo|image|picture|graphic|illustration)\s+(showing|of|depicting|with)?\s*",
        r"^(an?\s+)?close\s*up\s+(photo|image|picture|shot)?\s*(of|showing)?\s*",
        r"^(an?\s+)?stock\s+photo\s+(of|showing)?\s*",
        r"^(there\s+is|there\s+are)\s*",
    ]
    
    for pat in filler_patterns:
        cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE).strip()

    # Clean leading punctuation
    cleaned = re.sub(r"^[,\.\-\:\;\s]+", "", cleaned).strip()

    return cleaned.capitalize()


class AdvancedVisionAnalyzer:
    """
    Advanced & Hyper-Accurate Image-to-Text Neural Vision Engine.
    Uses Salesforce BLIP-Large with 10-Beam Search Decoding and Microsoft GIT with strict token cleaning.
    Zero OCR, Zero '[ unused0 ]' artifacts.
    """
    def __init__(self):
        self._git_processor = None
        self._git_model = None
        self._blip_processor = None
        self._blip_model = None
        self._init_attempted = False

    @staticmethod
    def safe_imread(path: str):
        """Robust OpenCV imread for Windows supporting non-ASCII paths and spaces."""
        try:
            data = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is not None:
                return img
        except Exception:
            pass
        return cv2.imread(path)

    def _init_engines(self):
        """Initialize Salesforce BLIP-Large and Microsoft GIT neural models lazily."""
        if self._init_attempted:
            return
        self._init_attempted = True

        # 1. Initialize Salesforce BLIP Large (Primary SOTA Vision Model)
        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            logger.info("Loading Primary SOTA Vision Model ('Salesforce/blip-image-captioning-large')...")
            self._blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
            self._blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
            self._blip_model.eval()
            logger.info("Salesforce BLIP Large loaded successfully!")
        except Exception as e:
            logger.warning(f"Failed BLIP-Large ({e}). Trying 'Salesforce/blip-image-captioning-base'...")
            try:
                from transformers import BlipProcessor, BlipForConditionalGeneration
                fallback = "Salesforce/blip-image-captioning-base"
                self._blip_processor = BlipProcessor.from_pretrained(fallback)
                self._blip_model = BlipForConditionalGeneration.from_pretrained(fallback)
                self._blip_model.eval()
            except Exception as ex:
                logger.error(f"BLIP Model load error: {ex}")
                self._blip_processor = None
                self._blip_model = None

        # 2. Initialize Microsoft GIT Model (Secondary Vision Model)
        try:
            from transformers import AutoProcessor, AutoModelForCausalLM
            logger.info("Loading Secondary Vision Model ('microsoft/git-base')...")
            self._git_processor = AutoProcessor.from_pretrained("microsoft/git-base")
            self._git_model = AutoModelForCausalLM.from_pretrained("microsoft/git-base")
            self._git_model.eval()
            logger.info("Microsoft GIT Base loaded successfully!")
        except Exception as e:
            logger.warning(f"GIT Model load deferred: {e}")
            self._git_processor = None
            self._git_model = None

    def analyze_visual_features(self, image_path: str) -> Dict[str, Any]:
        """Analyzes image visual attributes: color palette, brightness, contrast, framing, edge complexity."""
        abs_path = os.path.abspath(os.path.normpath(image_path))
        if not os.path.exists(abs_path):
            return {"prompt": "Image file not found", "style_tags": []}

        img = self.safe_imread(abs_path)
        if img is None:
            return {"prompt": "Invalid image format", "style_tags": []}

        h, w, c = img.shape
        aspect = w / h if h > 0 else 1.0

        # Brightness & Contrast
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))

        # Color Palette
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        avg_color = np.mean(img_rgb, axis=(0, 1))
        r, g, b = avg_color[0], avg_color[1], avg_color[2]

        color_name = "balanced palette"
        if r > g + 25 and r > b + 25:
            color_name = "warm red and amber tones"
        elif g > r + 25 and g > b + 25:
            color_name = "lush green nature tones"
        elif b > r + 25 and b > g + 25:
            color_name = "cool blue ocean tones"
        elif r > 180 and g > 180 and b < 100:
            color_name = "golden hour lighting"
        elif brightness < 60:
            color_name = "dark moody shadows"
        elif brightness > 200:
            color_name = "bright high-key lighting"

        # Framing
        if aspect > 1.6:
            shot_type = "widescreen framing"
        elif aspect < 0.8:
            shot_type = "vertical framing"
        else:
            shot_type = "standard aspect framing"

        # Edge Complexity
        edges = cv2.Canny(gray, 100, 200)
        edge_density = float(np.mean(edges))
        detail_desc = "intricate detailed texture" if edge_density > 25 else "clean minimalist composition"

        style_tags = [color_name, shot_type, detail_desc]
        
        prompt = f"{shot_type} with {color_name} and {detail_desc}"
        return {
            "prompt": prompt,
            "style_tags": style_tags,
            "brightness": brightness,
            "contrast": contrast,
            "aspect_ratio": aspect
        }

    def generate_image_to_text(self, image_path: str, detailed: bool = True) -> Dict[str, Any]:
        """
        Generates High-Accuracy Image-to-Text Description & Reverse Prompt.
        
        :param image_path: Absolute path to PNG image file.
        :param detailed: Include visual style tags.
        :return: Dict containing 'prompt', 'base_caption', and 'style_tags'.
        """
        abs_path = os.path.abspath(os.path.normpath(image_path))
        if not os.path.exists(abs_path):
            return {"prompt": "Image not found", "base_caption": "", "style_tags": []}

        self._init_engines()

        captions_candidates = []

        if os.path.exists(abs_path):
            try:
                raw_image = Image.open(abs_path).convert('RGB')

                # 1. Salesforce BLIP Large Engine (Primary SOTA captioner with 10-beam search)
                if self._blip_model is not None and self._blip_processor is not None:
                    try:
                        # Unconditional Pass
                        inputs_blip1 = self._blip_processor(raw_image, return_tensors="pt")
                        out_blip1 = self._blip_model.generate(
                            **inputs_blip1,
                            max_new_tokens=150,
                            min_length=15,
                            num_beams=10,
                            no_repeat_ngram_size=3,
                            repetition_penalty=1.3
                        )
                        blip_caption1 = self._blip_processor.decode(out_blip1[0], skip_special_tokens=True).strip()
                        cleaned_blip1 = clean_to_the_point_text(blip_caption1)
                        if cleaned_blip1 and len(cleaned_blip1) > 5:
                            captions_candidates.append(cleaned_blip1)

                        # Detailed Scene Pass
                        inputs_blip2 = self._blip_processor(raw_image, text="a detailed photograph showing", return_tensors="pt")
                        out_blip2 = self._blip_model.generate(
                            **inputs_blip2,
                            max_new_tokens=150,
                            min_length=15,
                            num_beams=10,
                            no_repeat_ngram_size=3,
                            repetition_penalty=1.3
                        )
                        blip_caption2 = self._blip_processor.decode(out_blip2[0], skip_special_tokens=True).strip()
                        cleaned_blip2 = clean_to_the_point_text(blip_caption2)
                        if cleaned_blip2 and len(cleaned_blip2) > 5:
                            captions_candidates.append(cleaned_blip2)

                    except Exception as e:
                        logger.error(f"BLIP Large error on {abs_path}: {e}")

                # 2. Microsoft GIT Engine (Filtered)
                if self._git_model is not None and self._git_processor is not None:
                    try:
                        inputs_git = self._git_processor(images=raw_image, return_tensors="pt")
                        pixel_values = inputs_git.pixel_values
                        out_git = self._git_model.generate(
                            pixel_values=pixel_values,
                            max_new_tokens=100,
                            num_beams=5,
                            repetition_penalty=1.2
                        )
                        git_caption = self._git_processor.batch_decode(out_git, skip_special_tokens=True)[0].strip()
                        cleaned_git = clean_to_the_point_text(git_caption)
                        if cleaned_git and len(cleaned_git) > 5 and "unused" not in git_caption.lower():
                            captions_candidates.append(cleaned_git)
                    except Exception as e:
                        logger.error(f"Microsoft GIT error on {abs_path}: {e}")

            except Exception as e:
                logger.error(f"Image open error on {abs_path}: {e}")

        # 3. Computer Vision Attributes
        visual_info = self.analyze_visual_features(abs_path)

        # Select richest, most descriptive candidate
        if captions_candidates:
            primary_caption = max(captions_candidates, key=len)
            primary_caption = clean_to_the_point_text(primary_caption)
        else:
            primary_caption = clean_to_the_point_text(visual_info["prompt"])

        # Build SOTA Neural Description Prompt
        prompt_parts = [primary_caption]

        if detailed and visual_info.get("style_tags"):
            tags_str = ", ".join(visual_info["style_tags"])
            prompt_parts.append(f"-- style: {tags_str}")

        full_prompt = " ".join(prompt_parts)

        return {
            "prompt": full_prompt,
            "base_caption": primary_caption,
            "style_tags": visual_info.get("style_tags", [])
        }

    def process_frame_list(
        self,
        frames_info: List[Dict[str, Any]],
        detailed_prompts: bool = True,
        progress_callback: Optional[Callable[[int, int, Dict[str, Any]], None]] = None,
        stop_checker: Optional[Callable[[], bool]] = None
    ) -> List[Dict[str, Any]]:
        """
        Processes all frame images sequentially from frame 1 to last frame.
        Stores generated image-to-text records for all frames in an array.
        """
        prompts_array = []
        total = len(frames_info)

        for idx, frame in enumerate(frames_info, start=1):
            if stop_checker and stop_checker():
                break

            img_path = frame.get("image_path", "")
            res = self.generate_image_to_text(img_path, detailed=detailed_prompts)

            record = {
                "frame_index": frame.get("frame_index", idx),
                "timestamp_sec": frame.get("timestamp_sec", 0.0),
                "filename": frame.get("filename", os.path.basename(img_path)),
                "image_path": img_path,
                "image_prompt": res["prompt"],
                "base_caption": res["base_caption"],
                "style_tags": res.get("style_tags", [])
            }

            prompts_array.append(record)

            if progress_callback:
                progress_callback(idx, total, record)

        return prompts_array


# Backward compatibility aliases
BestImageToTextEngine = AdvancedVisionAnalyzer
TopClassImageAnalyzer = AdvancedVisionAnalyzer
VisionEngine = AdvancedVisionAnalyzer
