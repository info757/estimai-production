"""
Test script for processing a single sheet (e.g., C-2.1) from Dawn Ridge PDF.

Finds the page containing the specified sheet number, processes only that page,
and generates accuracy results.
"""

import os
import sys
import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.vision import UniversalVisionAgent, parse_markdown_to_json
from app.vision.text_based_extract import extract_sewer_pipes
from app.evaluation.custom_metrics import evaluate_takeoff_custom
import fitz  # PyMuPDF for PDF text extraction

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_sheet_page(pdf_path: str, sheet_number: str) -> Optional[int]:
    """
    Find the page number (0-indexed) containing the specified sheet number.
    
    This function distinguishes between:
    - Index pages (contain many sheet codes, e.g., page 2)
    - Actual drawing pages (contain the sheet code in title block + drawing content)
    
    Args:
        pdf_path: Path to PDF file
        sheet_number: Sheet number to find (e.g., "C-2.1")
        
    Returns:
        Page number (0-indexed) if found, None otherwise
    """
    import re
    
    logger.info(f"Searching for sheet '{sheet_number}' in {pdf_path}")
    
    doc = fitz.open(pdf_path)
    
    # First, try to parse sheet index to get page mapping (if index exists)
    # Look for index page (has many C- codes)
    index_page = None
    ordered_sheet_codes: List[str] = []
    for page_num in range(min(5, len(doc))):
        page = doc[page_num]
        text = page.get_text()
        c_code_matches = re.findall(r'C-\d+[\.-]?\d*', text, re.IGNORECASE)
        if len(c_code_matches) > 10:  # Index pages have many sheet codes
            index_page = page_num
            logger.info(f"Found sheet index on page {page_num + 1}")

            seen_codes = set()
            for raw_code in c_code_matches:
                normalized = raw_code.upper().replace(' ', '').replace('--', '-').replace('-.', '-')
                if normalized not in seen_codes:
                    seen_codes.add(normalized)
                    ordered_sheet_codes.append(normalized)
            break
    
    # Now search for actual drawing page with this sheet number
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        text_upper = text.upper()
        
        # Skip index pages
        if page_num == index_page:
            continue
        
        # Look for sheet number in title block area (first 800 chars typically)
        title_area = text_upper[:800]
        
        search_patterns = [
            sheet_number.upper(),
            sheet_number.replace("-", " ").upper(),  # "C 2.1"
            sheet_number.replace("-", "").upper(),   # "C2.1"
        ]
        
        found_in_title = False
        for pattern in search_patterns:
            if pattern in title_area:
                found_in_title = True
                break
        
        if not found_in_title:
            continue
        
        # Verify this is not just an index page - check for drawing content
        # Drawing pages have specific indicators
        has_drawing_indicators = any([
            bool(re.search(r'\d+\s*LF', text_upper)),  # Pipe callouts with lengths
            'PVC' in text_upper or 'DIP' in text_upper,  # Material indicators
            'PROFILE' in text_upper or 'INVERT' in text_upper,  # Profile indicators
            bool(re.search(r'\d+\.\d+\s*FT', text_upper)),  # Elevations
        ])
        
        # Index pages have many sheet codes, drawing pages have few
        c_code_count = len(re.findall(r'C-\d+\.?\d*', text, re.IGNORECASE))
        is_index_page = c_code_count > 10
        
        if has_drawing_indicators and not is_index_page:
            has_lf = bool(re.search(r'\d+\s*LF', text_upper))
            has_material = bool('PVC' in text_upper or 'DIP' in text_upper)
            logger.info(f"Found sheet '{sheet_number}' on page {page_num + 1} (0-indexed: {page_num})")
            logger.info(f"  Drawing indicators: LF={has_lf}, Material={has_material}, C-codes={c_code_count}")
            doc.close()
            return page_num
    
    # Fallback: use index order mapping if direct detection failed
    target_code = sheet_number.upper()
    if ordered_sheet_codes and target_code in ordered_sheet_codes and index_page is not None:
        position = ordered_sheet_codes.index(target_code)
        estimated_page_idx = index_page + 1 + position  # Pages after index follow order
        if estimated_page_idx < len(doc):
            logger.info(
                "Direct lookup failed; using index mapping to estimate page %s (0-indexed: %s)",
                estimated_page_idx + 1,
                estimated_page_idx,
            )
            doc.close()
            return estimated_page_idx

    doc.close()
    logger.warning(f"Sheet '{sheet_number}' not found in PDF")
    return None


async def test_single_sheet(
    pdf_path: str,
    sheet_number: str,
    firm: str = "hagen_engineering",
    use_three_pass: bool = True,
    use_text_extraction: bool = True,
) -> Dict[str, Any]:
    """Test extraction on a single sheet/page."""

    logger.info("=" * 80)
    logger.info(f"TESTING SINGLE SHEET: {sheet_number}")
    logger.info(f"Extraction Mode: {'Text-based (vector + OCR)' if use_text_extraction else 'Vision LLM'}")
    logger.info("=" * 80)

    page_index = find_sheet_page(pdf_path, sheet_number)
    if page_index is None:
        raise ValueError(f"Sheet '{sheet_number}' not found in PDF")

    page_number = page_index + 1  # Convert to 1-indexed
    logger.info(f"Processing page {page_number} (0-indexed: {page_index})")

    start_time = datetime.now()

    predicted_data: Dict[str, Any]
    markdown: str
    firm_detected = firm
    metadata: Dict[str, Any] = {}

    if use_text_extraction:
        logger.info("\nRunning text-based extraction (vector text + OCR)...")
        extraction_results = extract_sewer_pipes(pdf_path, page_number)

        aggregated_pipes = extraction_results.get("pipes", [])

        predicted_data = {
            "pipes": [
                {
                    "diameter_in": pipe.get("diameter_in"),
                    "material": pipe.get("material", "Unknown"),
                    "length_ft": pipe.get("length_ft", 0.0),
                    "count": pipe.get("count", 1),
                    "type": "Pipe",
                    "discipline": "sanitary",
                }
                for pipe in aggregated_pipes
            ],
            "structures": [],
            "earthwork": [],
            "metadata": {
                "pages_processed": 1,
                "total_pipes": len(aggregated_pipes),
                "total_structures": 0,
                "total_earthwork": 0,
                "firm": firm,
                "extraction_source": extraction_results.get("source", "unknown"),
            },
        }

        metadata = predicted_data["metadata"]

        markdown_lines = [
            "# Construction Document Extraction\n\n",
            "**Extraction Method**: Text-based (vector + OCR)\n",
            f"**Source**: {extraction_results.get('source', 'unknown')}\n\n",
            "## Pipes\n\n",
        ]

        if aggregated_pipes:
            for idx, pipe in enumerate(aggregated_pipes, start=1):
                markdown_lines.extend(
                    [
                        f"### Pipe {idx}\n",
                        f"- Diameter: {pipe.get('diameter_in', 'N/A')} inches\n",
                        f"- Material: {pipe.get('material', 'Unknown')}\n",
                        f"- Length: {pipe.get('length_ft', 0):.2f} LF\n",
                        f"- Count: {pipe.get('count', 1)}\n\n",
                    ]
                )
        else:
            markdown_lines.append("No pipes detected.\n")

        markdown = "".join(markdown_lines)
    else:
        logger.info("\nInitializing Universal Vision Agent...")
        agent = UniversalVisionAgent()

        logger.info(f"\nAnalyzing page {page_number} with sheet {sheet_number}...")
        llm_results = await agent.analyze_document(
            pdf_path=pdf_path,
            firm=firm,
            auto_detect_firm=True,
            use_three_pass=use_three_pass,
            page_range=[page_number],
        )

        markdown = llm_results["markdown"]
        firm_detected = llm_results.get("firm_detected", firm)

        logger.info("\nParsing markdown to structured JSON...")
        predicted_data = parse_markdown_to_json(markdown)
        metadata = llm_results.get("metadata", predicted_data.get("metadata", {}))

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    logger.info(f"\nExtraction completed in {duration:.1f} seconds")
    logger.info(f"  - Pipes found: {len(predicted_data.get('pipes', []))}")

    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)

    markdown_path = output_dir / f"sheet_{sheet_number.replace('-', '_')}_extraction.md"
    with open(markdown_path, "w") as f:
        f.write(markdown)
    logger.info(f"\nSaved markdown extraction to: {markdown_path}")

    json_path = output_dir / f"sheet_{sheet_number.replace('-', '_')}_extraction.json"
    with open(json_path, "w") as f:
        json.dump(predicted_data, f, indent=2)
    logger.info(f"Saved parsed JSON to: {json_path}")

    total_lf = sum(p.get("length_ft", 0) * p.get("count", 1) for p in predicted_data.get("pipes", []))

    report = f"""# Single Sheet Test Report: {sheet_number}

**Date**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Sheet Number**: {sheet_number}
**Page Number**: {page_number}
**Firm**: {firm_detected}
**Method**: {'Text-based (vector + OCR)' if use_text_extraction else 'Vision LLM'}
**Processing Time**: {duration:.1f} seconds

---

## Extraction Results

### Pipes
- **Count**: {len(predicted_data.get('pipes', []))}
- **Total Linear Feet**: {total_lf:.2f} LF

### Structures
- **Count**: {len(predicted_data.get('structures', []))}

### Earthwork
- **Count**: {len(predicted_data.get('earthwork', []))}

---

## Extracted Pipes

"""

    for idx, pipe in enumerate(predicted_data.get("pipes", [])[:10], start=1):
        report += (
            f"{idx}. {pipe.get('diameter_in', 'N/A')}\" {pipe.get('material', 'N/A')}"
            f" - {pipe.get('length_ft', 0):.2f} LF\n"
        )

    extra_pipes = len(predicted_data.get("pipes", [])) - 10
    if extra_pipes > 0:
        report += f"\n... and {extra_pipes} more pipes\n"

    report_path = output_dir / f"sheet_{sheet_number.replace('-', '_')}_report.md"
    with open(report_path, "w") as f:
        f.write(report)
    logger.info(f"Saved report to: {report_path}")

    logger.info("\n" + "=" * 80)
    logger.info("TEST COMPLETE")
    logger.info("=" * 80)

    return {
        "sheet_number": sheet_number,
        "page_number": page_number,
        "results": predicted_data,
        "metadata": metadata,
        "duration": duration,
        "markdown": markdown,
        "output_files": {
            "markdown": str(markdown_path),
            "json": str(json_path),
            "report": str(report_path),
        },
    }


# =============================================================================
# Main Execution
# =============================================================================

async def main():
    """Main entry point."""
    raw_args = sys.argv[1:]

    use_text = "--llm" not in raw_args
    positional_args = [arg for arg in raw_args if not arg.startswith("--")]

    sheet_number = positional_args[0] if positional_args else "C-2.1"

    base_dir = Path(__file__).parent.parent
    pdf_path = base_dir / "data/test_pdfs/Dawn Ridge Homes_HEPA_Combined_04-1-25.pdf"

    if not pdf_path.exists():
        logger.error(f"PDF not found: {pdf_path}")
        return

    try:
        results = await test_single_sheet(
            pdf_path=str(pdf_path),
            sheet_number=sheet_number,
            firm="hagen_engineering",
            use_three_pass=True,
            use_text_extraction=use_text,
        )
        
        logger.info(f"\n✅ Successfully processed sheet {sheet_number}")
        logger.info(f"📄 Results saved to: {results['output_files']['report']}")
        
    except Exception as e:
        logger.error(f"Error testing sheet {sheet_number}: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())
