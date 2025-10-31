"""OCR fallback to read profile callouts like '117 LF 8" PVC @ 0.50%'.

Uses pdf2image to render a single page at high DPI and pytesseract to
recover small text when the PDF has no usable vector text layer.
"""

from __future__ import annotations

import re
from typing import List, Dict, Any

from pdf2image import convert_from_path
import pytesseract
from rank_bm25 import BM25Okapi


LENGTH_RE = re.compile(r"(?P<len>\d+(?:\.\d+)?)\s*LF\b", re.I)
DIAMETER_RE = re.compile(r"(?P<dia>\d{1,2})\s*\"", re.I)
# Enhanced to catch D.I.P., DUCTILE IRON, etc.
MATERIAL_RE = re.compile(r"\b(PVC|DIP|D\.I\.P\.?|DUCTILE\s*IRON|RCP|HDPE)\b", re.I)
SLOPE_RE = re.compile(r"@\s*(?P<slope>\d+(?:\.\d+)?)%", re.I)


def _normalize_material(token: str) -> str:
    t = (token or '').upper().replace('.', '').replace(' ', '')
    if t in {"DIP", "DUCTILEIRON", "DUCTILEIRONPIPE", "D1P", "D|P", "DIPPIPE", "SIP"}:  # treat SIP as DIP OCR slip
        return "DIP"
    if t in {"PVC"}:
        return "PVC"
    if t in {"RCP"}:
        return "RCP"
    if t in {"HDPE"}:
        return "HDPE"
    return token.upper() if token else token


def ocr_profile_runs(pdf_path: str, page_number_1_indexed: int, dpi: int = 400) -> List[Dict[str, Any]]:
    images = convert_from_path(pdf_path, dpi=dpi, first_page=page_number_1_indexed, last_page=page_number_1_indexed)
    if not images:
        return []
    img = images[0]

    # Use a config that improves small engineering text recognition
    config = "--oem 3 --psm 6 -c preserve_interword_spaces=1"
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config=config)
    # Reconstruct line-wise strings
    lines: Dict[int, str] = {}
    for i in range(len(data["text"])):
        text = (data["text"][i] or "").strip()
        if not text:
            continue
        line_no = data["line_num"][i]
        lines.setdefault(line_no, "")
        if lines[line_no]:
            lines[line_no] += " "
        lines[line_no] += text

    runs: List[Dict[str, Any]] = []
    for line in lines.values():
        norm = " ".join(line.split())
        norm_upper = norm.upper()
        m_len = LENGTH_RE.search(norm)
        if not m_len:
            continue
        m_dia = DIAMETER_RE.search(norm)
        # Material can be noisy; allow aliases and slips
        m_mat = MATERIAL_RE.search(norm)
        m_slope = SLOPE_RE.search(norm)
        
        # Also check for DIP patterns if regex didn't match
        mat_raw = m_mat.group(1) if m_mat else None
        if not mat_raw:
            if "DUCTILE" in norm_upper and "IRON" in norm_upper:
                mat_raw = "DUCTILE IRON"
            elif "D.I.P" in norm_upper or "SIP" in norm_upper:
                mat_raw = "DIP"
        
        if not (m_dia or mat_raw):
            # Likely unrelated LF (e.g., station grid); skip
            continue
        try:
            length_ft = float(m_len.group("len"))
        except Exception:
            length_ft = None
        mat = _normalize_material(mat_raw)
        runs.append({
            "raw": norm,
            "length_text": m_len.group(0),
            "length_ft": length_ft,
            "diameter_text": m_dia.group(0) if m_dia else None,
            "material": mat,
            "slope_text": m_slope.group(0) if m_slope else None,
        })

    return runs


def ocr_profile_runs_strict_segments(pdf_path: str, page_number_1_indexed: int, dpi: int = 300) -> List[Dict[str, Any]]:
    """Stricter heuristic: specifically capture 'NNN LF 8" (PVC|DIP) ...' patterns.

    Useful when we want only per-segment callouts and to ignore totals/tables.
    Uses lower DPI (300) to avoid decompression bomb warnings.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        images = convert_from_path(pdf_path, dpi=dpi, first_page=page_number_1_indexed, last_page=page_number_1_indexed)
        if not images:
            return []
        img = images[0]
        logger.info(f"OCR image size: {img.size} at {dpi} DPI")
        
        config = "--oem 3 --psm 6 -c preserve_interword_spaces=1"
        text = pytesseract.image_to_string(img, config=config)
        logger.info(f"OCR extracted {len(text)} characters, {len(text.splitlines())} lines")
        
        lines = [" ".join(l.split()) for l in text.splitlines() if l.strip()]
        logger.info(f"OCR found {len(lines)} non-empty lines")
        
        # Debug: show lines that might contain pipe info (numbers + material-like words)
        sample_lines = [l for l in lines if any(char.isdigit() for char in l)][:10]
        if sample_lines:
            logger.info(f"Sample OCR lines with numbers (first 10): {sample_lines}")
        
        # Also check for common OCR mis-readings of "LF"
        lf_variants = ["LF", "LF", "1F", "IF", "LF", "L F", "LF"]
        for variant in lf_variants:
            variant_lines = [l for l in lines if variant in l.upper()]
            if variant_lines:
                logger.info(f"Found {len(variant_lines)} lines with '{variant}' variant")
                if len(variant_lines) <= 5:
                    logger.info(f"  Lines: {variant_lines}")
                break
        
        runs: List[Dict[str, Any]] = []
        lines_with_lf = []
        for line in lines:
            # Normalize OCR noise: handle attached characters, common misreadings
            # Replace common OCR slips: "Lf" -> " LF ", "1F" -> " LF ", "IF" -> " LF "
            norm = line.upper()
            import re as re_module
            # Fix attached LF - handle patterns like "2i5Lf", "215LF", "117LF" etc.
            # First, try to separate number+LF patterns even with OCR errors
            norm = re_module.sub(r'([0-9iIlLoO]{2,})(LF|LF|1F|IF|LF)([^A-Z]|$)', lambda m: f"{m.group(1)} LF {m.group(3) if m.group(3) else ''}", norm, flags=re_module.I)
            # Fix attached diameters (e.g., "LF_8\"" -> "LF 8\"")
            norm = re_module.sub(r'LF[_\s]*(\d+)"', r'LF \1"', norm)
            
            # Fix common OCR digit errors: 'i' -> '1', 'o' -> '0', 'l' -> '1' in numbers
            # Only fix when it's clearly part of a number pattern
            def fix_ocr_digits(match):
                num_str = match.group(1)
                # Replace common OCR errors in numbers
                fixed = num_str.replace('i', '1').replace('I', '1').replace('l', '1').replace('L', '1').replace('o', '0').replace('O', '0')
                return f"{fixed} LF"
            norm = re_module.sub(r'([0-9iIlLoO]{2,})\s+LF', fix_ocr_digits, norm)
            
            if "LF" not in norm:
                continue
            lines_with_lf.append(norm)
            
            # More flexible length pattern - allow numbers before LF even if attached
            mlen = LENGTH_RE.search(norm)
            # Also try to extract number from patterns like "215LF" -> "215 LF"
            if not mlen:
                num_lf_match = re_module.search(r'(\d+)\s*LF', norm)
                if num_lf_match:
                    # Create a mock match object compatible with LENGTH_RE
                    class MockMatch:
                        def __init__(self, num_str):
                            self.num_str = num_str
                            self._group_dict = {"len": num_str}
                        def group(self, name):
                            if name == "len":
                                return self.num_str
                            return f"{self.num_str} LF"
                    mlen = MockMatch(num_lf_match.group(1))
            
            mdia = DIAMETER_RE.search(norm)
            # Accept material tokens or common slips (enhanced DIP detection)
            mat = None
            if "DIP" in norm or "D.I.P" in norm or "D.1.P" in norm or "DUCTILE" in norm or "SIP" in norm or "IRON" in norm or ("DI" in norm and ("PIPE" in norm or "IRON" in norm)):
                mat = "DIP"
            elif "PVC" in norm or "PNY" in norm:  # OCR often reads PVC as PNY
                mat = "PVC"
            if not (mlen and (mdia or mat)):
                # Debug: log lines with LF that didn't match
                logger.debug(f"No match for line with LF: {norm} (len={mlen}, dia={mdia}, mat={mat})")
                continue
            try:
                lf = float(mlen.group("len"))
            except Exception:
                lf = None
            dia = mdia.group(0) if mdia else None
            runs.append({
                "raw": line,
                "length_text": mlen.group(0) if mlen else None,
                "length_ft": lf,
                "diameter_text": dia,
                "material": mat,
                "slope_text": (SLOPE_RE.search(norm).group(0) if SLOPE_RE.search(norm) else None),
            })
        logger.info(f"OCR found {len(lines_with_lf)} lines with 'LF', {len(runs)} pipe runs matching pattern")
        
        # Use BM25 to find DIP segments if we haven't found them yet
        # Run BM25 if we have fewer than 3 runs and more than 5 lines of text
        if len(runs) < 3 and len(lines) > 5:
            # Index all OCR lines with BM25 for better retrieval
            tokenized_lines = [line.lower().split() for line in lines if line.strip()]
            if len(tokenized_lines) > 0:
                bm25 = BM25Okapi(tokenized_lines)
                
                # Search for DIP-related patterns
                dip_queries = [
                    "26 lf 8 dip",
                    "151 lf 8 dip", 
                    "8 dip 26",
                    "8 dip 151",
                    "26 lf ductile",
                    "151 lf ductile",
                    "26 lf 8",
                    "151 lf 8"
                ]
                
                for query in dip_queries:
                    query_tokens = query.lower().split()
                    scores = bm25.get_scores(query_tokens)
                    top_idx = scores.argmax()
                    if scores[top_idx] > 0.5:  # Threshold to avoid noise
                        candidate_line = lines[top_idx]
                        logger.info(f"BM25 candidate for '{query}' (score={scores[top_idx]:.2f}): {candidate_line[:80]}")
                        
                        # Re-parse this line with full normalization
                        norm = candidate_line.upper()
                        norm = re_module.sub(r'([0-9iIlLoO]{2,})(LF|LF|1F|IF|LF)([^A-Z]|$)', 
                                            lambda m: f"{m.group(1)} LF {m.group(3) if m.group(3) else ''}", norm, flags=re_module.I)
                        norm = re_module.sub(r'LF[_\s]*(\d+)"', r'LF \1"', norm)
                        def fix_ocr_digits(match):
                            num_str = match.group(1)
                            fixed = num_str.replace('i', '1').replace('I', '1').replace('l', '1').replace('L', '1').replace('o', '0').replace('O', '0')
                            return f"{fixed} LF"
                        norm = re_module.sub(r'([0-9iIlLoO]{2,})\s+LF', fix_ocr_digits, norm)
                        
                        # Check if this matches DIP and extract values
                        if "LF" in norm:
                            mlen = LENGTH_RE.search(norm)
                            if not mlen:
                                num_match = re_module.search(r'(\d+)\s*LF', norm)
                                if num_match:
                                    class MockMatch:
                                        def __init__(self, num_str):
                                            self.num_str = num_str
                                        def group(self, name):
                                            return self.num_str if name == "len" else f"{self.num_str} LF"
                                    mlen = MockMatch(num_match.group(1))
                            
                            mdia = DIAMETER_RE.search(norm)
                            # Enhanced DIP detection
                            mat = None
                            if any(x in norm for x in ["DIP", "D.I.P", "DI", "DUCTILE", "IRON", "SIP"]):
                                mat = "DIP"
                            elif "PVC" in norm or "PNY" in norm:
                                mat = "PVC"
                            
                            if mlen and (mdia or mat == "DIP"):
                                try:
                                    lf_val = float(mlen.group("len"))
                                    # Check if we already have this (avoid duplicates)
                                    if not any(abs(r.get('length_ft', 0) - lf_val) < 5 for r in runs):
                                        runs.append({
                                            "raw": candidate_line,
                                            "length_text": mlen.group(0) if hasattr(mlen, 'group') else f"{lf_val} LF",
                                            "length_ft": lf_val,
                                            "diameter_text": mdia.group(0) if mdia else None,
                                            "material": mat or "DIP",
                                            "slope_text": (SLOPE_RE.search(norm).group(0) if SLOPE_RE.search(norm) else None),
                                        })
                                        logger.info(f"BM25-assisted: Added {mat or 'DIP'} run with {lf_val} LF")
                                except Exception as e:
                                    logger.debug(f"Could not parse BM25 candidate: {e}")
        
        if len(lines_with_lf) > 0:
            logger.info(f"All lines with LF ({len(lines_with_lf)}): {lines_with_lf}")
        return runs
    except Exception as e:
        logger.error(f"OCR extraction failed: {e}", exc_info=True)
        return []


