#!/usr/bin/env python3
"""Run extraction on selected pages to test duplicate handling quickly."""

import asyncio
import logging
import sys
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.vision import UniversalVisionAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def run_selected_pages_test():
    """Run extraction on selected pages only."""
    logger.info("="*80)
    logger.info("SELECTED PAGES TEST: Dawn Ridge Homes")
    logger.info("="*80)

    # Paths
    base_dir = Path(__file__).parent.parent
    pdf_path = base_dir / "data/test_pdfs/Dawn Ridge Homes_HEPA_Combined_04-1-25.pdf"
    output_dir = base_dir / "results"
    output_dir.mkdir(exist_ok=True)

    # Pages to test (plan/profile/detail likely in early pages)
    test_pages = [1, 2, 3]
    logger.info(f"Pages selected: {test_pages}")

    # Initialize Universal Vision Agent
    logger.info("\nInitializing Universal Vision Agent...")
    agent = UniversalVisionAgent()

    # Run analysis
    logger.info(f"\nAnalyzing PDF: {pdf_path.name} (Pages {test_pages})")
    logger.info("This should take ~5-6 minutes total...")

    start_time = datetime.now()

    results = await agent.analyze_document(
        pdf_path=str(pdf_path),
        firm="hagen_engineering",
        auto_detect_firm=True,
        use_three_pass=True,
        page_range=test_pages,
    )

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    # Save natural language output
    output_path = output_dir / "selected_pages_extraction.txt"
    with open(output_path, "w") as f:
        f.write(results["markdown"])

    logger.info("\n" + "="*80)
    logger.info("SELECTED PAGES TEST COMPLETE")
    logger.info("="*80)
    logger.info(f"Processing Time: {duration:.1f} seconds ({duration/60:.1f} minutes)")
    logger.info(f"Output saved to: {output_path}")

    return results


if __name__ == "__main__":
    asyncio.run(run_selected_pages_test())
