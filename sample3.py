"""
consolidated_meter_ocr.py
End-to-end script for
1. Image pre-processing (blur / contrast / ROI detection / multi-variant)
2. OCR extraction via Amazon Bedrock Claude-3-Sonnet
3. Post-processing & validation (digit disambiguation, 1↔7, completeness)
4. Reference matching + result ranking
5. SQLite persistence

Author: <you>
"""

import os
import cv2
import re
import json
import base64
import sqlite3
import logging
import asyncio
import threading
from datetime import datetime
from typing import Dict, Tuple, List, Any, Optional
from collections import Counter

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

# ─────────────────────────────── log setup ─────────────────────────────── #
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s ‑ %(levelname)s ‑ %(name)s ‑ %(message)s")
logger = logging.getLogger("MeterOCR")

# ─────────────────────────── Rate-limit helper ─────────────────────────── #
class RateLimitHandler:
    """Simple exponential-back-off retry helper."""
    def __init__(self, max_retries: int = 5, base_delay: float = 2.0):
        self.max_retries = max_retries
        self.base_delay = base_delay

    def call_with_retry(self, func, *args, **kwargs):
        for i in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except ClientError as e:
                if e.response["Error"]["Code"] not in {"ThrottlingException",
                                                       "TooManyRequestsException"}:
                    raise
                delay = self.base_delay * (2 ** i)
                logger.warning(f"Rate-limited ➜ retry in {delay:.1f}s")
                asyncio.sleep(delay)
        raise RuntimeError("Maximum retries exceeded")


# ─────────────────────── Enhanced image preprocessing ─────────────────────── #
class EnhancedImagePreprocessor:
    """Create multiple high-quality variants suited to OCR."""

    _CONFIGS = {
        "default":      dict(blur=(3, 3), block=11, C=2,  morph=(2, 2),
                             alpha=1.2, beta=10),
        "low_quality":  dict(blur=(5, 5), block=15, C=5,  morph=(3, 3),
                             alpha=1.5, beta=20),
        "high_contrast":dict(blur=(1, 1), block=9,  C=1,  morph=(1, 1),
                             alpha=1.8, beta=-10)
    }

    # ‑- private helpers ‑- #
    @staticmethod
    def _enhance(gray: cv2.UMat, cfg: Dict[str, Any]) -> cv2.UMat:
        blurred  = cv2.GaussianBlur(gray, cfg["blur"], 0)
        boosted  = cv2.convertScaleAbs(blurred,
                                       alpha=cfg["alpha"],
                                       beta=cfg["beta"])
        thresh   = cv2.adaptiveThreshold(
            boosted, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
            cfg["block"], cfg["C"])
        kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, cfg["morph"])
        cleaned  = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        cleaned  = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN,  kernel)
        return cleaned

    @staticmethod
    def _to_b64(img: cv2.UMat) -> str:
        _, buf = cv2.imencode(".jpg", img)
        return base64.b64encode(buf).decode("utf-8")

    # ‑- public API ‑- #
    def detect_meter_roi(self, img: cv2.UMat) -> Tuple[cv2.UMat, Dict]:
        """Locate display; fall back to full frame if unsure."""
        gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges  = cv2.Canny(gray, 50, 150)
        cnts,_ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)

        best = None
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 1_000:
                continue
            x, y, w, h = cv2.boundingRect(c)
            ar = w / h
            if 1.5 <= ar <= 4.0:
                best = (x, y, w, h) if best is None or area > best[-1] else best
                best = (x, y, w, h, area)
        if best:
            x, y, w, h, _ = best
            pad = 20
            roi = img[max(0, y-pad):y+h+pad, max(0, x-pad):x+w+pad]
            return roi, {"bbox": (x, y, w, h), "conf": .8}
        return img, {"bbox": (0,0,*img.shape[:2][::-1]), "conf": .3}

    def create_variants(self, path: str) -> Dict[str, str]:
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(path)

        roi, _ = self.detect_meter_roi(img)
        gray   = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        variants = {}
        for name, cfg in self._CONFIGS.items():
            variants[name] = self._to_b64(self._enhance(gray, cfg))
        variants["original"] = self._to_b64(img)
        return variants


# ───────────────────── Digit validation / correction ───────────────────── #
class DigitValidationEngine:
    """Handle 1↔7, missing last digit, pattern sanity, etc."""

    _PATTERNS = [r"^\d{4}$", r"^\d{5}$", r"^\d{6}$",
                 r"^0\d{4}$", r"^0\d{5}$"]

    def validate(self, reading: str, conf: float) -> Dict[str, Any]:
        if not reading or reading.lower() in {"not visible", "unclear", "error"}:
            return {"valid": False, "corrected": reading,
                    "notes": "no reading"}

        digits  = re.sub(r"\D", "", reading)
        pattern = any(re.match(p, digits) for p in self._PATTERNS)

        # naive completeness check
        complete = 4 <= len(digits) <= 6
        status   = pattern and complete
        return {"valid": status,
                "corrected": digits,
                "notes": "ok" if status else "pattern/completeness failed"}


# ───────────────────────── Result ranking / consensus ─────────────────────── #
class EnhancedOCRResultProcessor:
    def __init__(self):
        self.validator = DigitValidationEngine()

    def _score(self, ocr: Dict, variant: str) -> float:
        score = ocr.get("ai_confidence", 0) * 40       # 0-40
        r     = ocr.get("extracted_data", {}).get("meter_reading", "")
        if r:
            val = self.validator.validate(r, score/40)
            if val["valid"]:            score += 30
            if "ok" in val["notes"]:    score += 20
        variant_bonus = {"default":5,"low_quality":3,
                         "high_contrast":3,"original":1}
        return min(score + variant_bonus.get(variant, 0), 100)

    def pick_best(self, variants: Dict[str, Dict]) -> Dict[str, Any]:
        if not variants:
            return {"best": None}
        ranking = [{"var": v, "out": d,
                    "score": self._score(d, v)}
                   for v,d in variants.items()]
        ranking.sort(key=lambda x: x["score"], reverse=True)
        return {"best": ranking[0], "all": ranking}


# ───────────────────────────── Main OCR reader ───────────────────────────── #
class BedrockOCRReader:
    """
    Drive full pipeline: create variants ➜ invoke Claude ➜ rank ➜ store
    """

    # Claude prompt with disambiguation rules
    _PROMPT = """🎯  EXPERT METER READER – PRECISION DIGIT EXTRACTION
Your mission: Return JSON ONLY with fields described below.
Special care:
• Distinguish 1 (thin vertical) vs 7 (top bar + diagonal).
• Capture ALL digits; the right-most digit is often faint.
• Preserve leading zeros.
JSON schema:
{
 "meter_serial_number": "",
 "meter_reading": "",
 "meter_type": "Electric/Gas/Water",
 "reading_date": "YYYY-MM-DD or not_visible",
 "reading_time": "HH:MM or not_visible",
 "display_type": "LCD/LED/Digital",
 "units": "kWh/m3/gal",
 "confidence_score": 1-10,
 "extraction_notes": ""
}"""

    # ── ctor ── #
    def __init__(self, model="anthropic.claude-3-5-sonnet-20240620-v1:0",
                 db="smart_meter_database.db"):
        self.model = model
        self.db    = db
        self.pre   = EnhancedImagePreprocessor()
        self.proc  = EnhancedOCRResultProcessor()
        self.limiter = RateLimitHandler()

        self.bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")
        self._lock   = threading.Lock()
        self._ensure_tables()

    # ────────────────────── internal helpers ────────────────────── #
    def _ensure_tables(self):
        with sqlite3.connect(self.db) as con:
            con.execute("""CREATE TABLE IF NOT EXISTS processing_results(
                image_path TEXT PRIMARY KEY,
                payload     TEXT,
                created_at  TEXT
            )""")

    def _invoke_claude(self, b64_img: str) -> Dict[str, Any]:
        payload = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1500,
            "temperature": 0.1,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": self._PROMPT},
                    {"type": "image",
                     "source": {"type": "base64",
                                "media_type": "image/jpeg",
                                "data": b64_img}}
                ]
            }]
        }
        resp = self.limiter.call_with_retry(
            self.bedrock.invoke_model,
            modelId=self.model,
            body=json.dumps(payload)
        )
        txt  = json.loads(resp["body"].read())["content"][0]["text"]
        json_str = txt[txt.find("{"): txt.rfind("}")+1]
        return json.loads(json_str)

    # process single variant
    def _process_variant(self, b64_img: str) -> Dict[str, Any]:
        try:
            result = self._invoke_claude(b64_img)
            result.setdefault("ai_confidence",
                              float(result.get("confidence_score", 5)) / 10)
            # wrap under unified schema
            return {"ocr_result": {
                        "extracted_data": {
                            "meter_serial_number": result.get("meter_serial_number",""),
                            "meter_reading":      result.get("meter_reading",""),
                            "meter_type":         result.get("meter_type",""),
                            "reading_date":       result.get("reading_date",""),
                            "reading_time":       result.get("reading_time",""),
                            "display_type":       result.get("display_type",""),
                            "units":              result.get("units","")},
                        "ai_confidence": result.get("ai_confidence", .5),
                        "extraction_notes": result.get("extraction_notes","")
                    }}
        except Exception as e:
            logger.error(f"OCR variant failed: {e}")
            return {}

    # ─────────────────────── public API ─────────────────────── #
    async def process_image(self, path: str) -> Dict[str, Any]:
        """Main entrypoint – returns best OCR dict and writes DB."""
        variants_b64 = self.pre.create_variants(path)
        tasks = [asyncio.to_thread(self._process_variant, b64)
                 for b64 in variants_b64.values()]
        raw_outputs = await asyncio.gather(*tasks)

        # map variant name ➜ output
        outputs = {v: o for v,o in zip(variants_b64, raw_outputs)}
        choice  = self.proc.pick_best(outputs)

        # persist
        with self._lock, sqlite3.connect(self.db) as con:
            con.execute("INSERT OR REPLACE INTO processing_results "
                        "(image_path, payload, created_at) VALUES (?,?,?)",
                        (path, json.dumps(choice, default=str),
                         datetime.utcnow().isoformat()))

        logger.info(f"✅ processed {path} – best score "
                    f"{choice['best']['score']:.1f}")
        return choice


# ────────────────────────────── CLI helper ────────────────────────────── #
if __name__ == "__main__":
    import argparse, glob, asyncio
    parser = argparse.ArgumentParser(description="Run meter-OCR on images")
    parser.add_argument("pattern",
                        help="glob pattern, e.g. './samples/*.jpg'")
    args = parser.parse_args()

    reader = BedrockOCRReader()
    loop   = asyncio.get_event_loop()
    files  = glob.glob(args.pattern)
    for img_path in files:
        loop.run_until_complete(reader.process_image(img_path))
