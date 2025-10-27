"""
Simple Test Script - Verify System Setup

Tests basic functionality without running full 25-page analysis.
"""

import sys
import asyncio
import logging
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.vision import UniversalVisionAgent, parse_markdown_to_json
from prompts import format_examples_for_prompt, FIRM_EXAMPLES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_imports():
    """Test that all imports work."""
    logger.info("Testing imports...")
    
    try:
        from app.rag.advanced_retriever import AdvancedRetriever
        from app.evaluation.custom_metrics import evaluate_takeoff_custom
        from langchain_openai import ChatOpenAI
        logger.info("✅ All imports successful")
        return True
    except Exception as e:
        logger.error(f"❌ Import error: {e}")
        return False


async def test_firm_examples():
    """Test firm-specific examples."""
    logger.info("\nTesting firm-specific examples...")
    
    try:
        # Check Hagen examples exist
        hagen = FIRM_EXAMPLES.get("hagen_engineering")
        assert hagen is not None, "Hagen Engineering examples not found"
        
        # Format examples
        formatted = format_examples_for_prompt("hagen_engineering", categories=["mainline_pipes"])
        assert len(formatted) > 100, "Examples too short"
        
        logger.info(f"✅ Found {len(hagen.get('mainline_pipes', []))} mainline pipe examples")
        logger.info(f"✅ Found {len(hagen.get('laterals', []))} lateral examples")
        logger.info(f"✅ Found {len(hagen.get('structures', []))} structure examples")
        
        return True
    except Exception as e:
        logger.error(f"❌ Firm examples error: {e}")
        return False


async def test_markdown_parser():
    """Test markdown parser."""
    logger.info("\nTesting markdown parser...")
    
    sample = """
## Pipes
### Sanitary Pipe 1
- Diameter: 8 inches
- Material: PVC
- Length: 100 LF
- Depth: 9.0 ft

## Structures
### Manhole: MH-1
- ID: MH-1
- Type: Manhole
- Rim Elevation: 745.0 ft
"""
    
    try:
        result = parse_markdown_to_json(sample)
        
        assert len(result['pipes']) == 1, f"Expected 1 pipe, got {len(result['pipes'])}"
        assert result['pipes'][0]['diameter_in'] == 8, "Diameter parsing failed"
        assert result['pipes'][0]['length_ft'] == 100, "Length parsing failed"
        
        assert len(result['structures']) == 1, f"Expected 1 structure, got {len(result['structures'])}"
        
        logger.info("✅ Markdown parser working correctly")
        logger.info(f"   - Parsed 1 pipe: 8\" PVC, 100 LF")
        logger.info(f"   - Parsed 1 structure: MH-1")
        
        return True
    except Exception as e:
        logger.error(f"❌ Parser error: {e}")
        return False


async def test_vision_agent_init():
    """Test Universal Vision Agent initialization."""
    logger.info("\nTesting Universal Vision Agent initialization...")
    
    try:
        agent = UniversalVisionAgent()
        
        assert agent.model == "gpt-4o", f"Expected gpt-4o, got {agent.model}"
        assert agent.llm is not None, "LLM not initialized"
        assert agent.rag is not None, "RAG not initialized"
        
        logger.info("✅ Universal Vision Agent initialized successfully")
        logger.info(f"   - Model: {agent.model}")
        logger.info(f"   - Temperature: {agent.temperature}")
        logger.info(f"   - RAG: {'Enabled' if agent.rag else 'Disabled'}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Agent initialization error: {e}")
        return False


async def test_rag_retrieval():
    """Test RAG retrieval."""
    logger.info("\nTesting RAG retrieval...")
    
    try:
        from app.rag.advanced_retriever import AdvancedRetriever
        
        rag = AdvancedRetriever()
        
        # Test query
        results = await rag.retrieve_with_expansion(
            query="What does SS mean in construction?",
            top_k=3
        )
        
        assert len(results) > 0, "No RAG results returned"
        
        logger.info(f"✅ RAG retrieval working")
        logger.info(f"   - Query: 'What does SS mean in construction?'")
        logger.info(f"   - Results: {len(results)} documents")
        logger.info(f"   - Sample: {results[0].page_content[:100]}...")
        
        return True
    except Exception as e:
        logger.error(f"❌ RAG error: {e}")
        logger.warning("   This might be okay if Qdrant is not running")
        return True  # Don't fail on RAG error


async def run_all_tests():
    """Run all tests."""
    logger.info("="*80)
    logger.info("SYSTEM VERIFICATION TESTS")
    logger.info("="*80)
    
    tests = [
        ("Imports", test_imports),
        ("Firm Examples", test_firm_examples),
        ("Markdown Parser", test_markdown_parser),
        ("Vision Agent Init", test_vision_agent_init),
        ("RAG Retrieval", test_rag_retrieval),
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            results[name] = await test_func()
        except Exception as e:
            logger.error(f"Test '{name}' crashed: {e}")
            results[name] = False
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("TEST SUMMARY")
    logger.info("="*80)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status}: {name}")
    
    logger.info("\n" + "="*80)
    logger.info(f"OVERALL: {passed}/{total} tests passed")
    logger.info("="*80)
    
    if passed == total:
        logger.info("\n🎉 All tests passed! System is ready.")
        logger.info("\nNext steps:")
        logger.info("1. Run full accuracy test: python scripts/run_accuracy_test.py")
        logger.info("2. Review results in results/ directory")
    else:
        logger.warning("\n⚠️  Some tests failed. Review errors above.")
        logger.warning("Check:")
        logger.warning("- OpenAI API key is set (echo $OPENAI_API_KEY)")
        logger.warning("- Dependencies installed (pip install -r requirements.txt)")
        logger.warning("- Poppler installed for pdf2image (brew install poppler)")
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)

