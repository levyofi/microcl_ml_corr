import os
import re
import glob

SCENARIO_DIRS = {
    1: "scenario_1_valley_single_logger",
    2: "scenario_2_beach_single_logger",
    3: "scenario_3_desert_single_logger",
    4: "scenario_4_beach_pooled",
    5: "scenario_5_beach_specialized",
    6: "scenario_6_desert_pooled",
    7: "scenario_7_desert_specialized"
}

def generate_prediction_script(scen_id, dir_name):
    hist_file = os.path.join("inst", "examples", dir_name, "create_histograms.R")
    if not os.path.exists(hist_file):
        return None
        
    with open(hist_file, 'r') as f:
        content = f.read()
    
    # Global replaces
    content = content.replace("create_histograms.R", "create_prediction_examples.R")
    content = content.replace("hist_panels", "excerpt_panels")
    content = content.replace("p_hist", "p_excerpt")
    content = content.replace("residual_histogram_", "prediction_examples_")
    content = content.replace("histograms created successfully", "prediction examples created successfully")
    
    # Specific logic replacements
    if scen_id == 1:
        old_loop = re.search(r'for \(i in seq_along\(tasks_s1\)\) \{.*\}', content, re.DOTALL)
        if old_loop:
            new_loop = """for (task in tasks_s1) {
  excerpt_panels[[task$name]] <- make_pred_plot(head(full_dfs[[task$name]], 120), task$title, show_legend = TRUE)
}"""
            content = content.replace(old_loop.group(0), new_loop)
            
    elif scen_id == 2:
        content = re.sub(
            r'make_residual_hist\(full_df,\s*([^,]+),\s*show_legend\s*=\s*TRUE\)',
            r'make_pred_plot(head(full_df, 120), \1, show_legend = TRUE)',
            content
        )
        
    elif scen_id == 3:
        content = re.sub(
            r'make_residual_hist\(full_df,\s*([^,]+),\s*show_legend\s*=\s*TRUE\)',
            r'make_pred_plot(head(full_df, 120), \1, show_legend = TRUE)',
            content
        )
        
    elif scen_id == 4:
        old_loop = re.search(r'xlim_site <- range\(.*?\).*?make_residual_hist\(.*?show_legend = TRUE\n\s*\)', content, re.DOTALL)
        if old_loop:
            new_loop = 'excerpt_panels[[site]] <- make_pred_plot(head(site_df, 120), site, show_legend = TRUE)'
            content = content.replace(old_loop.group(0), new_loop)

    elif scen_id == 5:
        old_loop = re.search(r'xlim_site <- range\(.*?\).*?make_residual_hist\(.*?show_legend = TRUE\n\s*\)', content, re.DOTALL)
        if old_loop:
            new_loop = 'excerpt_panels[[panel_key]] <- make_pred_plot(head(site_df, 120), site, show_legend = TRUE)'
            content = content.replace(old_loop.group(0), new_loop)

    elif scen_id == 6:
        old_loop = re.search(r'xlim_site <- range\(.*?\).*?make_residual_hist\(.*?show_legend = TRUE\n\s*\)', content, re.DOTALL)
        if old_loop:
            new_loop = 'excerpt_panels[[site]] <- make_pred_plot(head(full_df, 120), title_str, show_legend = TRUE)'
            content = content.replace(old_loop.group(0), new_loop)

    elif scen_id == 7:
        old_loop = re.search(r'xlim_site <- range\(.*?\).*?make_residual_hist\(.*?show_legend = TRUE\n\s*\)', content, re.DOTALL)
        if old_loop:
            new_loop = 'excerpt_panels[[site]] <- make_pred_plot(head(full_df, 120), panel_title, show_legend = TRUE)'
            content = content.replace(old_loop.group(0), new_loop)

    return content

for scen_id, dir_name in SCENARIO_DIRS.items():
    code = generate_prediction_script(scen_id, dir_name)
    if code:
        out_file = os.path.join("inst", "examples", dir_name, "create_prediction_examples.R")
        with open(out_file, "w") as f:
            f.write(code)
        print(f"Wrote {out_file}")
