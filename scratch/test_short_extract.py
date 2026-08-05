import os
import glob
import re

def extract_short_findings(s_dir):
    report_file = glob.glob(os.path.join(s_dir, "scenario_*_report.md"))
    if not report_file:
        return ""
    
    with open(report_file[0], 'r', encoding='utf-8') as f:
        content = f.read()
    
    match = re.search(r'##.*?(?:Key Takeaway|Key Findings|Conclusions).*?\n+(.*?)(?=\n## |\Z)', content, re.DOTALL | re.IGNORECASE)
    if match:
        text = match.group(1).strip()
        # Remove markdown bold, italic, and headers
        text = re.sub(r'#+', '', text)
        text = text.replace('**', '').replace('_', '').replace('\n', ' ')
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Split into sentences and take first 2
        sentences = [s.strip() for s in text.split('. ') if s.strip()]
        if not sentences:
            return ""
            
        short_text = ". ".join(sentences[:2])
        if not short_text.endswith('.'):
            short_text += "."
            
        return short_text
    return ""

base_dir = "inst/examples"
for s_num in range(1, 9):
    s_dirs = glob.glob(os.path.join(base_dir, f"scenario_{s_num}_*"))
    if s_dirs:
        findings = extract_short_findings(s_dirs[0])
        print(f"Scenario {s_num}: {findings}")
