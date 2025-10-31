#!/usr/bin/env python3
"""Minimal test: Send profile image directly to LLM to see if it can read pipe callouts.
This bypasses all the complex extraction pipeline.
"""

import asyncio
import base64
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def image_to_b64(image_path: Path) -> str:
    """Convert image file to base64 string."""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


async def test_profile_image():
    """Test LLM's ability to read pipe callouts from profile image."""
    logger.info("="*80)
    logger.info("MINIMAL PROFILE IMAGE TEST")
    logger.info("="*80)
    
    # Load the profile image
    profile_image_path = Path(__file__).parent.parent / "assets/fewshots/profile/sewer_profile_example.png"
    
    if not profile_image_path.exists():
        logger.error(f"Profile image not found at: {profile_image_path}")
        return
    
    logger.info(f"Loading profile image: {profile_image_path}")
    image_b64 = image_to_b64(profile_image_path)
    
    # Simple, direct prompt
    system_prompt = "You are a construction estimator reading a sewer profile drawing. Your job is to read the exact text from pipe callouts/labels."
    
    user_prompt = """You are looking at a sewer profile drawing. 

Your task: List EVERY pipe segment callout you can see in the image.

For each callout, provide:
1. Length (LF) - copy the exact number as printed (e.g., "117 LF", "26 LF", "151 LF")
2. Diameter (e.g., "8\"")
3. Material - copy exactly as shown (e.g., "PVC", "DIP", "Ductile Iron", "D.I.P.")

Important:
- Copy the EXACT values as printed - do NOT round, estimate, or infer
- List each callout separately - if you see "117 LF 8" PVC" and "26 LF 8" DIP", list them as two separate items
- Look for ALL materials - there may be both PVC and DIP visible
- If you see multiple segments with different lengths, list them all

Format your response as a simple list, one callout per line:
Example:
- 117 LF, 8", PVC
- 26 LF, 8", DIP
- 151 LF, 8", DIP

Now list all the pipe callouts you see:"""
    
    # Initialize LLM
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.0  # No creativity, just read what's there
    )
    
    logger.info("\nSending image to LLM with simple prompt...")
    
    # Create message with image
    content = [
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{image_b64}",
                "detail": "high"  # High detail for small text
            }
        },
        {
            "type": "text",
            "text": user_prompt
        }
    ]
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=content)
    ]
    
    # Get response
    response = await llm.ainvoke(messages)
    
    logger.info("\n" + "="*80)
    logger.info("LLM RESPONSE:")
    logger.info("="*80)
    print(response.content)
    logger.info("="*80)
    
    # Save to file
    output_path = Path(__file__).parent.parent / "results" / "profile_image_test.txt"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(response.content)
    
    logger.info(f"\nSaved output to: {output_path}")


if __name__ == "__main__":
    asyncio.run(test_profile_image())

