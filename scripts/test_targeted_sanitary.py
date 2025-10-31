#!/usr/bin/env python3
"""Find 'Sewer Profile' via sheet index and extract that single page.
Then aggregate sanitary runs locally and compare to ground truth without
leaking any answers to the LLM.
"""

import asyncio
import base64
import io
import logging
import re
import sys
import json
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
from pdf2image import convert_from_path

load_dotenv()

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.vision import UniversalVisionAgent
from app.vision.vector_extract import extract_profile_runs_from_text
from app.vision.ocr_extract import ocr_profile_runs_strict_segments

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def image_to_b64(pil_image) -> str:
    buf = io.BytesIO()
    pil_image.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def load_image_file_b64(path: Path) -> str:
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


async def get_sewer_profile_sheet_code(agent: UniversalVisionAgent, pdf_path: Path) -> str:
    """Read cover/index (first 3 pages) and return 'sheet_code|title' for Sewer Profile.
    Returns e.g., 'C-2.1|Sewer Profile'.
    """
    pages = convert_from_path(str(pdf_path), first_page=1, last_page=3)
    fewshot_index = Path(__file__).parent.parent / "assets/fewshots/index/index_example.png"
    index_fs_b64 = load_image_file_b64(fewshot_index) if fewshot_index.exists() else None
    user_prompt = (
        "You are a construction sitework estimator with a civil engineering degree analyzing Hagen Engineering drawings.\n"
        "Using the first image as an example of a sheet index, read the second image (candidate page).\n"
        "From the index table, find entries relevant to TASK='sanitary/sewer' (e.g., 'Sewer Profile').\n"
        "Return ONLY: sheet_code|title (copied as seen, e.g., C-2.1|SEWER PROFILE). If none on this page, return NONE."
    )
    for pil in pages:
        img_b64 = image_to_b64(pil)
        images = [img_b64] if not index_fs_b64 else [index_fs_b64, img_b64]
        resp = await agent._call_vision_llm(
            image_b64=images,
            system_prompt="You locate target sheet codes from a sheet index by visual similarity to the example.",
            user_prompt=user_prompt,
        )
        resp = resp.strip()
        if resp.upper() != "NONE" and "|" in resp:
            return resp
    raise RuntimeError("Could not find Sewer Profile sheet code from sheet index in first 3 pages.")


async def find_pdf_page_for_sheet(agent: UniversalVisionAgent, pdf_path: Path, sheet_code: str, sheet_title: str) -> int:
    """Scan pages to find the PDF page whose title block matches the given sheet code and title.
    Returns 1-based PDF page index.
    """
    # Scan a reasonable range (first 30 pages for now)
    pages = convert_from_path(str(pdf_path), first_page=1, last_page=30)
    fewshot_title = Path(__file__).parent.parent / "assets/fewshots/title/title_example.png"
    title_fs_b64 = load_image_file_b64(fewshot_title) if fewshot_title.exists() else None
    verify_prompt = (
        f"You are a construction sitework estimator with a civil engineering degree.\n"
        f"Given the first image as a title-block example, and the second image as a candidate page,\n"
        f"does the candidate page’s title block match sheet_code {sheet_code} exactly and title approximately (numeric suffix allowed)?\n"
        "Answer ONLY YES or NO."
    )
    for idx, pil in enumerate(pages, start=1):
        img_b64 = image_to_b64(pil)
        images = [img_b64] if not title_fs_b64 else [title_fs_b64, img_b64]
        resp = await agent._call_vision_llm(
            image_b64=images,
            system_prompt="You verify sheet title blocks by visual similarity and exact code matching.",
            user_prompt=verify_prompt,
        )
        if resp.strip().upper().startswith("YES"):
            return idx
    raise RuntimeError(f"Could not verify page for {sheet_code} {sheet_title} in first 30 pages.")


def parse_and_aggregate_sanitary(text: str) -> dict:
    """Parse extracted text for pipes and aggregate totals by diameter/material/discipline.
    Returns dict with per-run list and aggregates.
    """
    # Extract the Pipes block
    m2 = re.search(r'^## Pipes\n([\s\S]*?)(?:^## |\Z)', text, re.M)
    pipes_block = m2.group(1) if m2 else ''
    items = re.split(r'^###\s+', pipes_block, flags=re.M)
    runs = []
    for blk in items:
        blk = blk.strip()
        if not blk:
            continue
        def g(rx):
            mm = re.search(rx, blk, re.I)
            return mm.group(1).strip() if mm else None
        name = blk.splitlines()[0]
        discipline = g(r'-\s*Discipline:\s*([^\n]+)')
        if discipline and discipline.lower() != 'sanitary':
            continue
        diameter = g(r'-\s*Diameter:\s*([^\n]+)')
        material = g(r'-\s*Material:\s*([^\n]+)')
        from_ = g(r'-\s*From:\s*([^\n]+)')
        to_ = g(r'-\s*To:\s*([^\n]+)')
        length = g(r'-\s*Length:\s*([0-9.]+)\s*LF') or g(r'-\s*Length\s*\(total\):\s*([0-9.]+)\s*LF')
        invert_in = g(r'-\s*Invert In:\s*([^\n]+)')
        invert_out = g(r'-\s*Invert Out:\s*([^\n]+)')
        try:
            length_ft = float(length) if length else None
        except Exception:
            length_ft = None
        runs.append({
            'name': name,
            'diameter': diameter,
            'material': material,
            'from': from_,
            'to': to_,
            'length_ft': length_ft,
            'invert_in': invert_in,
            'invert_out': invert_out,
        })

    # Aggregates
    agg = {}
    for r in runs:
        key = ((r['diameter'] or '').strip(), (r['material'] or '').strip())
        if key not in agg:
            agg[key] = {'total_lf': 0.0, 'count': 0}
        if r['length_ft']:
            agg[key]['total_lf'] += r['length_ft']
        agg[key]['count'] += 1

    return {'runs': runs, 'aggregates': agg}


async def run_targeted_sanitary():
    logger.info("="*80)
    logger.info("TARGETED TEST: Sewer Profile via Sheet Index (Verified)")
    logger.info("="*80)

    base_dir = Path(__file__).parent.parent
    pdf_path = base_dir / "data/test_pdfs/Dawn Ridge Homes_HEPA_Combined_04-1-25.pdf"
    output_dir = base_dir / "results"
    output_dir.mkdir(exist_ok=True)

    # Ground truth JSON path (for local comparison only)
    gt_json_path = base_dir / "data/ground_truth/dawn_ridge_annotations.json"

    logger.info("\nInitializing Universal Vision Agent...")
    agent = UniversalVisionAgent()

    # Step 1: Get sheet code|title for Sewer Profile from index
    logger.info("\nReading sheet index to locate 'Sewer Profile' sheet code...")
    try:
        sheet_entry = await get_sewer_profile_sheet_code(agent, pdf_path)
    except Exception as e:
        logger.error(f"Index parse failed: {e}")
        return
    sheet_code, sheet_title = [s.strip() for s in sheet_entry.split('|', 1)]
    logger.info(f"Index found: {sheet_code} | {sheet_title}")

    # Step 2: Verify page by scanning title blocks
    logger.info("Verifying the PDF page for that sheet code/title...")
    try:
        target_page = await find_pdf_page_for_sheet(agent, pdf_path, sheet_code, sheet_title)
    except Exception as e:
        logger.error(f"Page verification failed: {e}")
        return
    logger.info(f"Verified Sewer Profile on PDF page {target_page}")

    # Step 3: Extract only that page
    logger.info("\nAnalyzing the verified Sewer Profile page...")
    start_time = datetime.now()
    results = await agent.analyze_document(
        pdf_path=str(pdf_path),
        firm="hagen_engineering",
        auto_detect_firm=True,
        use_three_pass=True,
        page_range=[target_page],
    )
    duration = (datetime.now() - start_time).total_seconds()

    # Save raw output
    output_path = output_dir / "sanitary_profile_extraction.txt"
    with open(output_path, "w") as f:
        f.write(results["markdown"])

    # Step 4: Vector-first parse (authoritative numbers)
    vec_runs = extract_profile_runs_from_text(str(pdf_path), target_page)
    logger.info(f"Vector extraction found {len(vec_runs)} runs")
    
    # Always run OCR as cross-check, not just when vector is empty
    ocr_runs = []
    try:
        # Use strict segments to focus on per-segment callouts
        ocr_runs = ocr_profile_runs_strict_segments(str(pdf_path), target_page, dpi=450)
        logger.info(f"OCR extraction found {len(ocr_runs)} runs")
    except Exception as e:
        logger.warning(f"OCR fallback failed: {e}")
    
    # Replace lengths in parsed aggregation where available
    parsed = parse_and_aggregate_sanitary(results["markdown"])
    
    # Recompute aggregates from vector runs for 8" PVC and 8" DIP
    extracted_8_pvc_vector = 0.0
    extracted_8_dip_vector = 0.0
    for vr in vec_runs:
        if vr.length_ft and (vr.diameter_text or '').startswith('8'):
            if vr.material == 'PVC':
                extracted_8_pvc_vector += vr.length_ft
            elif vr.material == 'DIP':
                extracted_8_dip_vector += vr.length_ft

    # Aggregate from OCR runs
    extracted_8_pvc_ocr = 0.0
    extracted_8_dip_ocr = 0.0
    for r in ocr_runs:
        if r.get('length_ft') and (r.get('diameter_text') or '').startswith('8'):
            mat = (r.get('material') or '').upper()
            if mat == 'PVC':
                extracted_8_pvc_ocr += float(r['length_ft'])
            elif mat == 'DIP':
                extracted_8_dip_ocr += float(r['length_ft'])

    # Ground truth local comparison (NO LLM exposure)
    gt_8_pvc_total = None
    try:
        with open(gt_json_path, 'r') as f:
            gt = json.load(f)
        for p in gt.get('expected_pipes', []):
            if str(p.get('diameter_in')) == '8' and (p.get('material') or '').upper() == 'PVC' and (p.get('discipline') or '').lower() == 'sanitary':
                gt_8_pvc_total = float(p.get('length_ft') or 0)
                break
    except Exception:
        pass

    # Compute extracted 8" PVC total from LLM parse (for comparison)
    extracted_8_pvc_llm = 0.0
    for (diam, mat), vals in parsed['aggregates'].items():
        if (diam or '').startswith('8') and (mat or '').upper() == 'PVC':
            extracted_8_pvc_llm += vals['total_lf']

    logger.info("\n" + "="*80)
    logger.info("TARGETED TEST COMPLETE")
    logger.info("="*80)
    logger.info(f"Sewer Profile sheet: {sheet_code} | {sheet_title}")
    logger.info(f"Sewer Profile PDF page: {target_page}")
    logger.info(f"Processing Time: {duration:.1f} seconds ({duration/60:.1f} minutes)")
    logger.info(f"Output saved to: {output_path}")

    # Print aggregates
    logger.info("\nAggregates (Sanitary):")
    for (diam, mat), vals in parsed['aggregates'].items():
        logger.info(f"  - {diam} {mat}: total {vals['total_lf']:.2f} LF across {vals['count']} runs")

    # Ground truth for DIP
    gt_8_dip_total = None
    try:
        with open(gt_json_path, 'r') as f:
            gt = json.load(f)
        for p in gt.get('expected_pipes', []):
            if str(p.get('diameter_in')) == '8' and (p.get('material') or '').upper() == 'DIP' and (p.get('discipline') or '').lower() == 'sanitary':
                gt_8_dip_total = float(p.get('length_ft') or 0)
                break
    except Exception:
        pass

    if gt_8_pvc_total is not None:
        logger.info(f"\n8\" PVC total LF (vector): {extracted_8_pvc_vector:.2f}")
        logger.info(f"8\" PVC total LF (OCR): {extracted_8_pvc_ocr:.2f}")
        logger.info(f"8\" PVC total LF (LLM parse): {extracted_8_pvc_llm:.2f}")
        logger.info(f"8\" PVC total LF (ground truth): {gt_8_pvc_total:.2f}")
    
    if gt_8_dip_total is not None:
        logger.info(f"\n8\" DIP total LF (vector): {extracted_8_dip_vector:.2f}")
        logger.info(f"8\" DIP total LF (OCR): {extracted_8_dip_ocr:.2f}")
        logger.info(f"8\" DIP total LF (ground truth): {gt_8_dip_total:.2f}")
        logger.info(f"8\" DIP target segments: 26 LF + 151 LF = 177 LF")

    # Print vector runs for transparency, highlighting DIP
    if vec_runs:
        logger.info("\nVector-detected runs (sanitary candidates):")
        for i, vr in enumerate(vec_runs, 1):
            marker = " ⭐ DIP" if vr.material == 'DIP' else ""
            logger.info(f"  {i}. {vr.raw}  [len={vr.length_text}] [dia={vr.diameter_text}] [mat={vr.material}]{marker} [slope={vr.slope_text}]")
    if ocr_runs:
        logger.info("\nOCR-detected runs (sanitary candidates):")
        for i, r in enumerate(ocr_runs, 1):
            mat = r.get('material', '')
            marker = " ⭐ DIP" if mat == 'DIP' else ""
            logger.info(f"  {i}. {r['raw']}  [len={r['length_text']}] [dia={r['diameter_text']}] [mat={mat}]{marker} [slope={r['slope_text']}]")


if __name__ == "__main__":
    asyncio.run(run_targeted_sanitary())
