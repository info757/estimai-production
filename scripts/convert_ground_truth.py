#!/usr/bin/env python3
"""Convert JSON ground truth to natural language format."""

import json
from pathlib import Path


def convert_json_to_natural_language(json_path: str, output_path: str):
    """Convert JSON ground truth to natural language format."""
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    output = []
    output.append("# Ground Truth - Dawn Ridge Homes")
    output.append("")
    
    # Pipes section
    output.append("## Pipes")
    output.append("")
    for i, pipe in enumerate(data.get('expected_pipes', []), 1):
        output.append(f"### Pipe {i}: {pipe['structure_name']}")
        output.append(f"- Diameter: {pipe['diameter_in']} inches")
        output.append(f"- Material: {pipe['material']}")
        output.append(f"- Discipline: {pipe['discipline']}")
        output.append(f"- Type: {pipe['type']}")
        output.append(f"- Length: {pipe['length_ft']} LF")
        output.append(f"- Depth: {pipe['depth_ft']} ft")
        output.append(f"- Count: {pipe['count']}")
        output.append("")
    
    output.append(f"**Total Pipes: {len(data.get('expected_pipes', []))}**")
    output.append("")
    
    # Materials section
    if 'expected_materials' in data:
        output.append("## Expected Materials")
        output.append("")
        for material in data.get('expected_materials', []):
            output.append(f"- {material}")
        output.append("")
    
    # Volumes section
    if 'expected_volumes' in data:
        output.append("## Expected Volumes")
        output.append("")
        for vol in data.get('expected_volumes', []):
            output.append(f"- {vol}")
        output.append("")
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(output))
    
    print(f"✅ Converted ground truth to: {output_path}")
    print(f"   Total expected pipes: {len(data.get('expected_pipes', []))}")
    return len(data.get('expected_pipes', []))


if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent
    json_path = base_dir / "data/ground_truth/dawn_ridge_annotations.json"
    output_path = base_dir / "data/ground_truth/dawn_ridge_ground_truth.txt"
    
    pipe_count = convert_json_to_natural_language(str(json_path), str(output_path))
    print(f"\nGround truth conversion complete!")
    print(f"Review the output at: {output_path}")

