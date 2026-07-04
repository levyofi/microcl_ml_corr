import os
import re
import glob

def md_table_to_latex(md_table):
    lines = md_table.strip().split('\n')
    if len(lines) < 3:
        return ""
    
    headers = [col.strip() for col in lines[0].split('|') if col.strip()]
    headers = [col.replace('%', '\\%').replace('_', '\\_').replace('#', '\\#').replace('&', '\\&') for col in headers]
    num_cols = len(headers)
    
    latex = "\\begin{table}[H]\n\\centering\n"
    latex += "\\begin{tabular}{" + "c" * num_cols + "}\n"
    latex += "\\hline\n"
    latex += " & ".join(headers) + " \\\\\n"
    latex += "\\hline\n"
    
    for line in lines[2:]:
        cols = [col.strip() for col in line.split('|') if col.strip()]
        if cols:
            # Handle some markdown formatting like **bold** -> \textbf{bold}
            cols = [re.sub(r'\*\*(.*?)\*\*', r'\\textbf{\1}', col) for col in cols]
            cols = [col.replace('%', '\\%').replace('_', '\\_').replace('#', '\\#').replace('&', '\\&') for col in cols]
            latex += " & ".join(cols) + " \\\\\n"
            
    latex += "\\hline\n"
    latex += "\\end{tabular}\n\\end{table}\n"
    return latex

def extract_findings(s_dir):
    report_file = glob.glob(os.path.join(s_dir, "scenario_*_report.md"))
    if not report_file:
        return ""
    
    with open(report_file[0], 'r', encoding='utf-8') as f:
        content = f.read()
    
    match = re.search(r'##.*?(?:Key Takeaway|Key Findings|Conclusions).*?\n+(.*?)(?=\n## |\Z)', content, re.DOTALL | re.IGNORECASE)
    if match:
        text = match.group(1).strip()
        text = re.sub(r'#+', '', text)
        text = text.replace('**', '').replace('_', '').replace('\n', ' ')
        text = re.sub(r'\s+', ' ', text).strip()
        
        sentences = [s.strip() for s in text.split('. ') if s.strip()]
        if not sentences:
            return ""
            
        short_text = ". ".join(sentences[:2])
        if not short_text.endswith('.'):
            short_text += "."
            
        short_text = short_text.replace('%', '\\%').replace('_', '\\_').replace('#', '\\#').replace('&', '\\&')
        return short_text
    return ""

def process_scenario(s_num, s_dir):
    report_file = glob.glob(os.path.join(s_dir, f"scenario_{s_num}_report.md"))
    if not report_file:
        return ""
    
    with open(report_file[0], 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract Title
    title_match = re.search(r'# (Scenario \d+:.*?)\n', content)
    title = title_match.group(1).replace('_', '\\_') if title_match else f"Scenario {s_num}"

    # Extract Description (everything between Title and first ##)
    desc_match = re.search(r'#.*?\n+(.*?)(?=\n## )', content, re.DOTALL)
    description = desc_match.group(1).strip() if desc_match else ""
    # escape special chars
    description = description.replace('%', '\\%').replace('_', '\\_').replace('#', '\\#').replace('&', '\\&')

    # Extract Sample Sizes Table
    sample_sizes = ""
    ss_match = re.search(r'## 1\. Training Sample Sizes\n+(.*?)(?=\n## |\n\n)', content, re.DOTALL)
    if ss_match:
        table_text = ss_match.group(1).strip()
        sample_sizes = md_table_to_latex(table_text)
        
    # Extract Improvement Results
    # This might be under "## 5. Aggregated Hourly Summary", "## 6. Performance at Full Training Data", etc.
    imp_match = re.search(r'## \d+\. (?:Performance at Full Training Data|Aggregated Hourly Summary|Cross-Location Performance|Zero-Shot Performance).*?\n+(.*?)(?=\n## |\n\n|$)', content, re.DOTALL)
    improvement_results = ""
    if imp_match:
        table_text = imp_match.group(1).strip()
        improvement_results = md_table_to_latex(table_text)

    # Find Images
    images = glob.glob(os.path.join(s_dir, "residual_histogram_*.jpg")) + glob.glob(os.path.join(s_dir, "prediction_examples_*.jpg"))
    images = sorted(images)

    # Build LaTeX
    latex = f"\\section{{{title}}}\n\n"
    latex += f"\\textbf{{Description:}} {description}\n\n"
    if sample_sizes:
        latex += "\\textbf{Sample Sizes:}\n" + sample_sizes + "\n\n"
    if improvement_results:
        latex += "\\textbf{Improvement Results:}\n" + improvement_results + "\n\n"
        
    findings = extract_findings(s_dir)
    if findings:
        findings = f" {findings}"
        
    for img in images:
        rel_path = os.path.relpath(img, "inst/examples")
        caption_base = os.path.basename(img).replace('_', '\\_').replace('.jpg', '')
        
        if "prediction\\_examples" in caption_base:
            details = "Time-series prediction examples comparing observed temperatures (black solid line) to the baseline NicheMapR model (red dashed line) and the ML-corrected models (Random Forest in green dotted line, LSTM in blue solid line, if applicable) over a 120-hour window."
        elif "residual\\_histogram" in caption_base:
            details = "Distribution of prediction errors (residuals) comparing the baseline NicheMapR model (red) to the ML-corrected models (Random Forest in green, LSTM in blue, if applicable). A narrower distribution centered at zero indicates better performance."
        else:
            details = ""
            
        pretty_title = caption_base.replace('\\_', ' ').title()
        
        latex += "\\clearpage\n"
        latex += "\\begin{figure}[H]\n\\centering\n"
        latex += f"\\includegraphics[width=\\textwidth,height=0.85\\textheight,keepaspectratio]{{{rel_path}}}\n"
        latex += f"\\caption{{\\textbf{{{pretty_title} for Scenario {s_num}.}} {details}{findings}}}\n"
        latex += "\\end{figure}\n\n"
        
    return latex

def main():
    base_dir = "inst/examples"
    latex_out = os.path.join(base_dir, "scenarios_summary.tex")
    
    latex_doc = "\\documentclass{article}\n"
    latex_doc += "\\usepackage[utf8]{inputenc}\n"
    latex_doc += "\\usepackage{graphicx}\n"
    latex_doc += "\\usepackage{float}\n"
    latex_doc += "\\usepackage{geometry}\n"
    latex_doc += "\\usepackage{setspace}\n"
    latex_doc += "\\geometry{a4paper, margin=1in}\n"
    latex_doc += "\\onehalfspacing\n"
    latex_doc += "\\begin{document}\n\n"
    
    latex_doc += "\\title{Microclimate ML Correction - Scenarios Summary}\n"
    latex_doc += "\\maketitle\n\n"
    
    scenario_dirs = sorted(glob.glob(os.path.join(base_dir, "scenario_*")))
    
    for s_dir in scenario_dirs:
        # Extract number from dirname
        match = re.search(r'scenario_(\d+)', os.path.basename(s_dir))
        if match:
            s_num = int(match.group(1))
            latex_doc += process_scenario(s_num, s_dir)
            latex_doc += "\\clearpage\n\n"

    latex_doc += "\\end{document}\n"
    
    with open(latex_out, 'w', encoding='utf-8') as f:
        f.write(latex_doc)
        
    print(f"Generated LaTeX summary at {latex_out}")

if __name__ == '__main__':
    main()
