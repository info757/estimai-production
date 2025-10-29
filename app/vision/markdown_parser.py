"""
Deterministic Markdown → JSON Parser

Converts the LLM's structured markdown output into JSON using regex patterns.
This avoids the "JSON problem" by letting the LLM write natural language/markdown,
then parsing it deterministically.

Patterns are tuned for the structured markdown format from Universal Vision Agent.
"""

import re
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class MarkdownParser:
    """Parse structured markdown into JSON."""
    
    def __init__(self):
        """Initialize parser."""
        pass
    
    def parse(self, markdown_text: str) -> Dict[str, Any]:
        """
        Parse markdown extraction into structured JSON.
        
        Args:
            markdown_text: Markdown output from Universal Vision Agent
            
        Returns:
            Structured JSON with pipes, structures, earthwork, etc.
        """
        try:
            result = {
                "pipes": self._extract_pipes(markdown_text),
                "structures": self._extract_structures(markdown_text),
                "earthwork": self._extract_earthwork(markdown_text),
                "metadata": self._extract_metadata(markdown_text)
            }
            
            logger.info(f"Parsed {len(result['pipes'])} pipes, {len(result['structures'])} structures, {len(result['earthwork'])} earthwork items")
            return result
        except Exception as e:
            logger.error(f"Parsing error: {e}")
            return {"error": str(e), "raw_markdown": markdown_text}
    
    def _extract_pipes(self, markdown: str) -> List[Dict[str, Any]]:
        """
        Extract all pipes from markdown.
        
        Expected format:
        ## Pipes
        ### [Discipline] Pipe N
        - Diameter: X inches
        - Material: PVC
        - Length: X LF
        ...
        """
        pipes = []
        
        # Find ## Pipes section
        pipes_section = self._extract_section(markdown, "Pipes")
        if not pipes_section:
            return pipes
        
        # Split into individual pipe blocks (### headers)
        pipe_blocks = re.split(r'###\s+', pipes_section)[1:]  # Skip first empty split
        logger.info(f"Found {len(pipe_blocks)} pipe blocks to parse")
        
        for block in pipe_blocks:
            try:
                pipe = self._parse_pipe_block(block)
                if pipe:
                    pipes.append(pipe)
            except Exception as e:
                logger.warning(f"Error parsing pipe block: {e}")
                continue
        
        return pipes
    
    def _parse_pipe_block(self, block: str) -> Optional[Dict[str, Any]]:
        """Parse individual pipe block."""
        pipe = {}
        
        # Extract discipline from header
        header_match = re.search(r'^(.*?)\s+Pipe', block, re.IGNORECASE)
        if header_match:
            pipe["discipline"] = header_match.group(1).strip().lower()
        
        # Extract fields
        pipe["diameter_in"] = self._extract_number(block, r'-\s*Diameter:\s*([\d.]+)\s*inch', default=0)
        pipe["material"] = self._extract_field(block, r'-\s*Material:\s*(\w+)', default="Unknown")
        pipe["type"] = self._extract_field(block, r'-\s*Type:\s*([\w\s]+)(?:\n|$)', default="Pipe")
        pipe["length_ft"] = self._extract_number(block, r'-\s*Length:\s*([\d.]+)\s*(?:LF|ft)', default=0)
        pipe["depth_ft"] = self._extract_number(block, r'-\s*Depth:\s*([\d.]+)\s*ft', default=0)
        
        # Extract from/to
        pipe["from"] = self._extract_field(block, r'-\s*From:\s*([^\n]+)', default="")
        pipe["to"] = self._extract_field(block, r'-\s*To:\s*([^\n]+)', default="")
        
        # Extract inverts
        pipe["invert_in"] = self._extract_number(block, r'-\s*Invert\s+In:\s*([\d.]+)\s*ft')
        pipe["invert_out"] = self._extract_number(block, r'-\s*Invert\s+Out:\s*([\d.]+)\s*ft')
        
        # Count
        pipe["count"] = self._extract_number(block, r'-\s*Count:\s*([\d.]+)', default=1.0)
        
        return pipe
    
    def _extract_structures(self, markdown: str) -> List[Dict[str, Any]]:
        """
        Extract all structures from markdown.
        
        Expected format:
        ## Structures
        ### [Type] N: [ID]
        - ID: MH-SS-1
        - Type: Manhole
        - Rim Elevation: X ft
        ...
        """
        structures = []
        
        # Find ## Structures section
        structures_section = self._extract_section(markdown, "Structures")
        if not structures_section:
            return structures
        
        # Split into individual structure blocks
        structure_blocks = re.split(r'###\s+', structures_section)[1:]
        
        for block in structure_blocks:
            try:
                structure = self._parse_structure_block(block)
                if structure:
                    structures.append(structure)
            except Exception as e:
                logger.warning(f"Error parsing structure block: {e}")
                continue
        
        return structures
    
    def _parse_structure_block(self, block: str) -> Optional[Dict[str, Any]]:
        """Parse individual structure block."""
        structure = {}
        
        # Extract ID and type from header
        header_match = re.search(r'^(.*?):\s*([^\n]+)', block)
        if header_match:
            structure["type"] = header_match.group(1).strip()
            structure["id"] = header_match.group(2).strip()
        
        # Extract fields
        structure["id"] = self._extract_field(block, r'-\s*ID:\s*([^\n]+)', default=structure.get("id", ""))
        structure["type"] = self._extract_field(block, r'-\s*Type:\s*([\w\s]+)(?:\n|$)', default=structure.get("type", "Unknown"))
        structure["discipline"] = self._extract_field(block, r'-\s*Discipline:\s*(\w+)', default="Unknown")
        
        # Extract elevations
        structure["rim_elevation"] = self._extract_number(block, r'-\s*Rim\s+Elevation:\s*([\d.]+)\s*ft')
        structure["invert_in"] = self._extract_number(block, r'-\s*Invert\s+In:\s*([\d.]+)\s*ft')
        structure["invert_out"] = self._extract_number(block, r'-\s*Invert\s+Out:\s*([\d.]+)\s*ft')
        structure["invert_elevation"] = self._extract_number(block, r'-\s*Invert\s+Elevation:\s*([\d.]+)\s*ft')
        structure["depth_ft"] = self._extract_number(block, r'-\s*Depth:\s*([\d.]+)\s*ft')
        
        return structure
    
    def _extract_earthwork(self, markdown: str) -> List[Dict[str, Any]]:
        """
        Extract earthwork items from markdown.
        
        Expected format:
        ## Earthwork
        ### Item N or ### Excavation N
        - Type: Site Grading
        - Volume: X cubic yards
        ...
        """
        earthwork_items = []
        
        # Find ## Earthwork section
        earthwork_section = self._extract_section(markdown, "Earthwork")
        if not earthwork_section:
            return earthwork_items
        
        # Split into individual blocks
        blocks = re.split(r'###\s+', earthwork_section)[1:]
        
        for block in blocks:
            try:
                item = self._parse_earthwork_block(block)
                if item:
                    earthwork_items.append(item)
            except Exception as e:
                logger.warning(f"Error parsing earthwork block: {e}")
                continue
        
        return earthwork_items
    
    def _parse_earthwork_block(self, block: str) -> Optional[Dict[str, Any]]:
        """Parse individual earthwork block."""
        item = {}
        
        # Extract fields
        item["type"] = self._extract_field(block, r'-\s*Type:\s*([\w\s/]+)(?:\n|$)', default="Unknown")
        item["volume_cy"] = self._extract_number(block, r'-\s*Volume:\s*([\d,]+)\s*(?:cubic yards|CY|cy)')
        item["depth_ft"] = self._extract_number(block, r'-\s*Depth:\s*([\d.]+)\s*ft')
        item["length_ft"] = self._extract_number(block, r'-\s*Length:\s*([\d.]+)\s*(?:LF|ft)')
        item["purpose"] = self._extract_field(block, r'-\s*Purpose:\s*([^\n]+)')
        
        # Extract cut/fill if present
        item["cut_cy"] = self._extract_number(block, r'-\s*Cut\s+Volume:\s*([\d,]+)\s*(?:cubic yards|CY|cy)')
        item["fill_cy"] = self._extract_number(block, r'-\s*Fill\s+Volume:\s*([\d,]+)\s*(?:cubic yards|CY|cy)')
        
        return item
    
    def _extract_metadata(self, markdown: str) -> Dict[str, Any]:
        """Extract document metadata."""
        metadata = {}
        
        # Extract page numbers mentioned
        page_pattern = r'#\s+Page\s+(\d+)'
        pages = re.findall(page_pattern, markdown)
        metadata["pages_processed"] = len(set(pages))
        
        # Extract totals from summary sections
        metadata["total_pipes"] = self._extract_number(markdown, r'-\s*Total\s+Pipes:\s*(\d+)', default=0)
        metadata["total_structures"] = self._extract_number(markdown, r'-\s*Total\s+Structures:\s*(\d+)', default=0)
        metadata["total_earthwork"] = self._extract_number(markdown, r'-\s*Total\s+Earthwork.*?:\s*(\d+)', default=0)
        
        # Extract firm
        metadata["firm"] = self._extract_field(markdown, r'\*\*Firm\*\*:\s*([^\n]+)')
        
        return metadata
    
    # =============================================================================
    # Helper Methods
    # =============================================================================
    
    def _extract_section(self, markdown: str, section_name: str) -> Optional[str]:
        """
        Extract a ## Section from markdown.
        
        Args:
            markdown: Full markdown text
            section_name: Section header name (without ##)
            
        Returns:
            Section content or None
        """
        # Match ## Section until next ## or end
        pattern = rf'##\s+{re.escape(section_name)}\s*\n(.*?)(?=\n##|\Z)'
        match = re.search(pattern, markdown, re.DOTALL | re.IGNORECASE)
        
        if match:
            return match.group(1).strip()
        return None
    
    def _extract_field(
        self,
        text: str,
        pattern: str,
        default: Any = None
    ) -> Any:
        """Extract text field using regex pattern."""
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return default
    
    def _extract_number(
        self,
        text: str,
        pattern: str,
        default: Optional[float] = None
    ) -> Optional[float]:
        """Extract numeric field using regex pattern."""
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            # Remove commas and convert to float
            num_str = match.group(1).replace(',', '')
            try:
                return float(num_str)
            except ValueError:
                return default
        return default


# =============================================================================
# Convenience Function
# =============================================================================

def parse_markdown_to_json(markdown_text: str) -> Dict[str, Any]:
    """
    Parse markdown extraction to JSON.
    
    Args:
        markdown_text: Markdown output from Universal Vision Agent
        
    Returns:
        Structured JSON dictionary
    """
    parser = MarkdownParser()
    return parser.parse(markdown_text)


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    # Test with sample markdown
    sample_markdown = """
# Page 1 Extraction

## Pipes
### Sanitary Pipe 1
- Diameter: 8 inches
- Material: PVC
- Discipline: Sanitary
- Type: Pipe
- From: MH-SS-1
- To: MH-SS-2
- Invert In: 742.5 ft
- Invert Out: 741.0 ft
- Length: 806.01 LF
- Depth: 9.0 ft

### Storm Pipe 1
- Diameter: 12 inches
- Material: RCP
- Discipline: Storm
- Type: Pipe
- Length: 150 LF
- Depth: 8.0 ft

## Structures
### Manhole: MH-SS-1
- ID: MH-SS-1
- Type: Manhole
- Discipline: Sanitary
- Rim Elevation: 745.0 ft
- Invert In: 742.5 ft
- Invert Out: 742.0 ft
- Depth: 3.0 ft

## Earthwork
### Site Grading
- Type: Site Grading
- Cut Volume: 1,234 cubic yards
- Fill Volume: 567 cubic yards

## Summary
- Total Pipes: 2
- Total Structures: 1
- Total Earthwork: 1
"""
    
    result = parse_markdown_to_json(sample_markdown)
    
    import json
    print(json.dumps(result, indent=2))

