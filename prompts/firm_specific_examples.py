"""
Firm-Specific Few-Shot Examples for Construction Document Extraction

This module contains notation patterns and examples from specific engineering firms
that the client works with repeatedly. System learns firm-specific conventions.
"""

from typing import Dict, List, Any, Optional

# =============================================================================
# HAGEN ENGINEERING (Primary Firm)
# Source: Dawn Ridge Homes_HEPA_Combined_04-1-25.pdf
# =============================================================================

HAGEN_ENGINEERING_EXAMPLES = {
    "firm_name": "Hagen Engineering",
    "detection_keywords": ["HAGEN ENGINEERING", "HAGEN"],
    
    "notation_guide": {
        # Utility abbreviations
        "sanitary": ["SS", "sanitary sewer", "sewer"],
        "storm": ["SD", "storm drain", "storm"],
        "water": ["WM", "water main", "water"],
        "gas": ["gas", "natural gas"],
        
        # Structure abbreviations
        "manhole": ["MH-SS-#", "MH-#", "manhole"],
        "catch_basin": ["CB-#", "catch basin"],
        "cleanout": ["CO", "cleanout", "SSL cleanout"],
        "valve": ["valve", "V"],
        "hydrant": ["FH", "fire hydrant"],
        
        # Elevation abbreviations
        "invert_elevation": ["IE", "invert"],
        "rim_elevation": ["RIM", "rim", "top"],
        "ground_level": ["GL", "ground"],
        "finished_grade": ["FG", "finish grade"],
        
        # Materials
        "pvc": ["PVC"],
        "dip": ["DIP", "ductile iron"],
        "hdpe": ["HDPE"],
        "rcp": ["RCP", "concrete pipe"],
        
        # Dimensions
        "diameter": ['"', "inch", "in"],
        "length": ["LF", "linear feet", "ft"],
        "depth": ["depth", "ft", "'"]
    },
    
    "mainline_pipes": [
        {
            "description": "8-inch PVC sanitary sewer mainline",
            "visual_notation": "8\" PVC SS from MH-SS-1 to MH-SS-2",
            "typical_location": "Profile view with invert elevations",
            "markdown_output": """## Sanitary Mainline Pipe
- Diameter: 8 inches
- Material: PVC
- Discipline: Sanitary
- Type: Pipe
- From: MH-SS-1
- To: MH-SS-2
- Invert In: 742.5 ft
- Invert Out: 741.0 ft
- Length: 806.01 LF
- Depth: 9.0 ft"""
        },
        {
            "description": "8-inch DIP sanitary sewer mainline",
            "visual_notation": "8\" DIP connecting to existing",
            "typical_location": "Plan view with connection detail",
            "markdown_output": """## Sanitary Mainline Pipe
- Diameter: 8 inches
- Material: DIP
- Discipline: Sanitary
- Type: Pipe
- Length: 177.67 LF
- Depth: 9.2 ft"""
        },
        {
            "description": "12-inch storm drain mainline",
            "visual_notation": "12\" SD from CB-1 to CB-2",
            "typical_location": "Storm drain profile",
            "markdown_output": """## Storm Drain Pipe
- Diameter: 12 inches
- Material: PVC
- Discipline: Storm
- Type: Pipe
- From: CB-1
- To: CB-2
- Invert In: 745.0 ft
- Invert Out: 744.5 ft
- Length: 150 LF
- Depth: 8.0 ft"""
        }
    ],
    
    "laterals": [
        {
            "description": "4-inch sanitary service laterals",
            "visual_notation": "4\" SS Service (26 ea)",
            "typical_location": "Plan view or legend/table",
            "markdown_output": """## Service Laterals
- Diameter: 4 inches
- Discipline: Sanitary
- Type: Lateral
- Count: 26
- Length: 816.96 LF (total for all)
- Average Depth: 6.1 ft"""
        },
        {
            "description": "6-inch storm drain laterals",
            "visual_notation": "6\" SD Lateral (8 ea)",
            "typical_location": "Plan view",
            "markdown_output": """## Storm Laterals
- Diameter: 6 inches
- Discipline: Storm
- Type: Lateral
- Count: 8
- Length: 240 LF (total)"""
        }
    ],
    
    "vertical_connections": [
        {
            "description": "Sanitary cleanouts with vertical riser",
            "visual_notation": "4\" SSL Cleanout (26 ea)",
            "typical_location": "Plan view or detail",
            "markdown_output": """## Vertical Cleanouts
- Diameter: 4 inches
- Discipline: Sanitary
- Type: Vertical
- Count: 26
- Depth: 4.0 ft"""
        }
    ],
    
    "structures": [
        {
            "description": "Sanitary manhole with inverts",
            "visual_notation": "MH-SS-1, RIM: 745.0, IE IN: 742.5, IE OUT: 742.0",
            "typical_location": "Plan view or profile",
            "markdown_output": """## Sanitary Manhole
- ID: MH-SS-1
- Type: Manhole
- Discipline: Sanitary
- Rim Elevation: 745.0 ft
- Invert In: 742.5 ft
- Invert Out: 742.0 ft
- Depth: 3.0 ft (RIM - IE)"""
        },
        {
            "description": "Storm catch basin",
            "visual_notation": "CB-1, RIM: 748.0, IE: 745.0",
            "typical_location": "Plan view",
            "markdown_output": """## Catch Basin
- ID: CB-1
- Type: Catch Basin
- Discipline: Storm
- Rim Elevation: 748.0 ft
- Invert Elevation: 745.0 ft
- Depth: 3.0 ft"""
        }
    ],
    
    "grading_and_excavation": [
        {
            "description": "Excavation volume for site grading",
            "visual_notation": "Cut: 1,234 CY, Fill: 567 CY",
            "typical_location": "Grading plan or volume table",
            "markdown_output": """## Earthwork
- Cut Volume: 1,234 cubic yards
- Fill Volume: 567 cubic yards
- Net Cut: 667 cubic yards
- Type: Site Grading"""
        },
        {
            "description": "Trench excavation for utilities",
            "visual_notation": "Trench excavation for 8\" SS, 9' depth",
            "typical_location": "Profile view or typical section",
            "markdown_output": """## Trench Excavation
- Purpose: 8-inch Sanitary Sewer
- Depth: 9.0 ft
- Width: 3.0 ft (typical)
- Length: 806 LF
- Volume: 269 cubic yards"""
        }
    ],
    
    "water_system": [
        {
            "description": "Water main with fire hydrants",
            "visual_notation": "8\" WM with FH",
            "typical_location": "Plan view",
            "markdown_output": """## Water Main
- Diameter: 8 inches
- Material: DIP
- Discipline: Water
- Type: Pipe
- Length: 500 LF
- Depth: 6.0 ft

## Fire Hydrants
- Type: Fire Hydrant
- Count: 3
- Connected to: 8\" Water Main"""
        }
    ]
}

# =============================================================================
# FIRM EXAMPLES REGISTRY
# =============================================================================

FIRM_EXAMPLES: Dict[str, Dict[str, Any]] = {
    "hagen_engineering": HAGEN_ENGINEERING_EXAMPLES,
    # Additional firms will be added as client encounters them
    # "firm_name_2": {...},
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def detect_firm_from_page(page_text: str) -> str:
    """
    Auto-detect engineering firm from page text (title block, logo, etc.).
    
    Args:
        page_text: Extracted text from page
        
    Returns:
        Firm identifier (e.g., "hagen_engineering") or "generic"
    """
    page_text_upper = page_text.upper()
    
    # Check each firm's detection keywords
    for firm_id, firm_data in FIRM_EXAMPLES.items():
        keywords = firm_data.get("detection_keywords", [])
        for keyword in keywords:
            if keyword.upper() in page_text_upper:
                return firm_id
    
    return "generic"


def get_firm_examples(firm_name: str, category: str = None) -> Any:
    """
    Get few-shot examples for specific firm and optional category.
    
    Args:
        firm_name: Firm identifier (e.g., "hagen_engineering")
        category: Optional category (e.g., "mainline_pipes", "laterals")
        
    Returns:
        Firm examples dict or specific category list
    """
    firm_data = FIRM_EXAMPLES.get(firm_name, {})
    
    if category:
        return firm_data.get(category, [])
    
    return firm_data


def get_notation_guide(firm_name: str) -> Dict[str, List[str]]:
    """
    Get notation/abbreviation guide for specific firm.
    
    Args:
        firm_name: Firm identifier
        
    Returns:
        Dictionary mapping standard terms to firm-specific notations
    """
    firm_data = FIRM_EXAMPLES.get(firm_name, {})
    return firm_data.get("notation_guide", {})


def format_examples_for_prompt(firm_name: str, categories: List[str] = None) -> str:
    """
    Format firm-specific examples into prompt-ready text.
    
    Args:
        firm_name: Firm identifier
        categories: Optional list of categories to include
        
    Returns:
        Formatted string for inclusion in LLM prompt
    """
    firm_data = FIRM_EXAMPLES.get(firm_name, {})
    
    if not firm_data:
        return "No firm-specific examples available."
    
    output = f"# {firm_data.get('firm_name', 'Unknown Firm')} Examples\n\n"
    
    # Add notation guide
    notation = firm_data.get("notation_guide", {})
    if notation:
        output += "## Common Abbreviations\n"
        for term, abbrevs in notation.items():
            output += f"- {term}: {', '.join(abbrevs)}\n"
        output += "\n"
    
    # Add examples for requested categories or all
    if not categories:
        categories = [k for k in firm_data.keys() 
                     if k not in ['firm_name', 'detection_keywords', 'notation_guide']]
    
    for category in categories:
        examples = firm_data.get(category, [])
        if examples:
            output += f"## {category.replace('_', ' ').title()}\n\n"
            for i, example in enumerate(examples, 1):
                output += f"### Example {i}: {example.get('description', 'N/A')}\n"
                output += f"**Visual Notation**: {example.get('visual_notation', 'N/A')}\n"
                output += f"**Typical Location**: {example.get('typical_location', 'N/A')}\n"
                output += f"**Expected Output**:\n{example.get('markdown_output', 'N/A')}\n\n"
    
    return output


def get_all_firm_names() -> List[str]:
    """Get list of all registered firm identifiers."""
    return list(FIRM_EXAMPLES.keys())


# =============================================================================
# LEARNING STRATEGY
# =============================================================================

def add_new_firm_examples(firm_id: str, firm_data: Dict[str, Any]) -> bool:
    """
    Add examples for a new engineering firm.
    
    Args:
        firm_id: Unique identifier for firm (e.g., "firm_name_engineering")
        firm_data: Dictionary containing firm examples (same structure as HAGEN_ENGINEERING_EXAMPLES)
        
    Returns:
        True if added successfully, False if already exists
    """
    if firm_id in FIRM_EXAMPLES:
        return False
    
    FIRM_EXAMPLES[firm_id] = firm_data
    return True


# For debugging/inspection
if __name__ == "__main__":
    print("Registered Firms:", get_all_firm_names())
    print("\n" + "="*80 + "\n")
    print(format_examples_for_prompt("hagen_engineering", categories=["mainline_pipes", "laterals"]))


