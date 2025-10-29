"""
Minimal Test - Just test PDF loading and basic functionality
"""

import asyncio
import logging
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.vision import UniversalVisionAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_minimal():
    """Test just PDF loading and basic agent initialization."""
    logger.info("Testing PDF loading...")
    
    pdf_path = Path(__file__).parent.parent / "data/test_pdfs/Dawn Ridge Homes_HEPA_Combined_04-1-25.pdf"
    
    if not pdf_path.exists():
        logger.error(f"PDF not found: {pdf_path}")
        return
    
    logger.info(f"PDF found: {pdf_path}")
    
    # Test PDF loading
    try:
        from pdf2image import convert_from_path
        pages = convert_from_path(str(pdf_path), dpi=150, first_page=1, last_page=1)
        logger.info(f"✅ PDF loaded successfully: {len(pages)} pages")
    except Exception as e:
        logger.error(f"❌ PDF loading failed: {e}")
        return
    
    # Test agent initialization
    try:
        agent = UniversalVisionAgent()
        logger.info("✅ Agent initialized successfully")
    except Exception as e:
        logger.error(f"❌ Agent initialization failed: {e}")
        return
    
    logger.info("✅ All basic tests passed!")

if __name__ == "__main__":
    asyncio.run(test_minimal())
