import re
import difflib
from typing import List, Dict, Any

class TextSummarizer:
    """
    Processes the frame image-to-text array, removes duplicates,
    and synthesizes a unified story AND a complete frame-by-frame detailed description listing.
    """
    def __init__(self, similarity_threshold: float = 0.65):
        self.similarity_threshold = similarity_threshold

    @staticmethod
    def _similarity(s1: str, s2: str) -> float:
        """Compute text similarity between two image descriptions."""
        if not s1 and not s2:
            return 1.0
        if not s1 or not s2:
            return 0.0
        return difflib.SequenceMatcher(None, s1.lower().strip(), s2.lower().strip()).ratio()

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        """Format seconds into MM:SS format."""
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins:02d}:{secs:02d}"

    def deduplicate_prompts_array(self, raw_prompts_array: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filters out consecutive and global duplicate frame descriptions.
        """
        if not raw_prompts_array:
            return []

        unique_events = []

        for item in raw_prompts_array:
            prompt = item.get("image_prompt", "").strip()
            base_cap = item.get("base_caption", prompt).strip()
            ts = item.get("timestamp_sec", 0.0)
            frame_idx = item.get("frame_index", 0)

            if not prompt or not base_cap:
                continue

            is_dup = False
            for existing in unique_events:
                sim = self._similarity(existing["base_caption"], base_cap)
                if sim >= self.similarity_threshold:
                    is_dup = True
                    if item.get("onscreen_text") and not existing.get("onscreen_text"):
                        existing["prompt"] = prompt
                        existing["onscreen_text"] = item.get("onscreen_text")
                    existing["end_frame"] = frame_idx
                    existing["end_time"] = ts
                    existing["frame_count"] += 1
                    break

            if not is_dup:
                unique_events.append({
                    "prompt": prompt,
                    "base_caption": base_cap,
                    "onscreen_text": item.get("onscreen_text", ""),
                    "start_frame": frame_idx,
                    "end_frame": frame_idx,
                    "start_time": ts,
                    "end_time": ts,
                    "frame_count": 1
                })

        return unique_events

    def generate_story_summary(
        self,
        raw_prompts_array: List[Dict[str, Any]],
        sentences_count: int = 5
    ) -> Dict[str, Any]:
        """
        Takes raw frame image-to-text array, generates ONE unified compact story summary,
        AND includes a complete frame-by-frame detailed description for every frame.
        
        :param raw_prompts_array: List of frame image prompt records.
        :param sentences_count: Target sentence count for summary.
        :return: Dict containing summary story text, markdown, and statistics.
        """
        dedup_events = self.deduplicate_prompts_array(raw_prompts_array)
        total_frames = len(raw_prompts_array)

        if not raw_prompts_array:
            return {
                "summary": "No visual scenes were identified from the video frames.",
                "story_markdown": "No visual scenes were identified from the video frames.",
                "dedup_events": [],
                "stats": {
                    "total_frames": 0,
                    "unique_scenes": 0,
                    "total_prompts": 0
                }
            }

        # Collect unique scene descriptions for overview
        unique_captions = []
        for event in dedup_events:
            cap = event["base_caption"]
            if event.get("onscreen_text"):
                cap += f" displaying text '{event['onscreen_text']}'"
            unique_captions.append(cap)

        full_narrative = ". ".join(unique_captions) + "."

        # Synthesize overview summary using sumy or fallback
        summary_sentences = []
        try:
            from sumy.parsers.plaintext import PlaintextParser
            from sumy.nlp.tokenizers import Tokenizer
            from sumy.summarizers.lsa import LsaSummarizer
            from sumy.nlp.stemmers import Stemmer
            from sumy.utils import get_stop_words

            parser = PlaintextParser.from_string(full_narrative, Tokenizer("english"))
            stemmer = Stemmer("english")
            summarizer = LsaSummarizer(stemmer)
            summarizer.stop_words = get_stop_words("english")

            sumy_results = summarizer(parser.document, max(2, min(sentences_count, len(dedup_events))))
            summary_sentences = [str(s) for s in sumy_results]
        except Exception:
            summary_sentences = unique_captions[:sentences_count]

        if not summary_sentences:
            summary_sentences = unique_captions[:sentences_count]

        unified_story_text = " ".join(summary_sentences)

        # Build clean Markdown output containing Overview AND Frame-by-Frame Detailed Descriptions
        story_md = []
        story_md.append("# Video Analysis & Detailed Frame Descriptions\n")
        story_md.append("### Unified Narrative Overview")
        story_md.append(unified_story_text + "\n")

        story_md.append("### Frame-by-Frame Detailed Descriptions")
        for item in raw_prompts_array:
            frame_num = item.get("frame_index", 1)
            ts_sec = item.get("timestamp_sec", 0.0)
            time_formatted = self._format_timestamp(ts_sec)
            prompt = item.get("image_prompt", "")
            fname = item.get("filename", "")
            story_md.append(f"- **Frame {frame_num}** `[{time_formatted} | {fname}]`: {prompt}")

        story_text = "\n".join(story_md)

        return {
            "summary": unified_story_text,
            "story_markdown": story_text,
            "dedup_events": dedup_events,
            "stats": {
                "total_frames": total_frames,
                "unique_scenes": len(dedup_events),
                "total_prompts": len(raw_prompts_array)
            }
        }
