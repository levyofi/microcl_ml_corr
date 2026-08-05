import os
import glob
import re

def extract_findings(s_dir):
    report_file = glob.glob(os.path.join(s_dir, "scenario_*_report.md"))
    if not report_file:
        return ""
    
    with open(report_file[0], 'r') as f:
        content = f.read()
    
    # Try to find 'Key Takeaway' or 'Key Findings'
    match = re.search(r'##.*?(?:Key Takeaway|Key Findings|Conclusions).*?\n+(.*?)(?=\n## |\Z)', content, re.DOTALL | re.IGNORECASE)
    if match:
        text = match.group(1).strip()
        # Clean up Markdown bold/italics
        text = text.replace('**', '').replace('_', '').replace('#', '').replace('\n', ' ')
        # Shorten if too long
        if len(text) > 400:
            text = text[:397] + "..."
        return text
    return ""

base_dir = "inst/examples"
for s_num in range(1, 9):
    s_dirs = glob.glob(os.path.join(base_dir, f"scenario_{s_num}_*"))
    if s_dirs:
        findings = extract_findings(s_dirs[0])
        print(f"Scenario {s_num} findings:")
        print(findings)
        print("-" * 50)
