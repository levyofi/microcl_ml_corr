import os

scen8_dir = "inst/examples/scenario_8_zero_shot_transfer"
hist_file = os.path.join(scen8_dir, "create_histograms.R")
out_file = os.path.join(scen8_dir, "create_prediction_examples.R")

with open(hist_file, "r") as f:
    content = f.read()

# Replace variables
content = content.replace("hist_panels", "excerpt_panels")
content = content.replace("p_arranged", "p_excerpt")
content = content.replace("residual_histogram_", "prediction_examples_")
content = content.replace("histograms created successfully", "prediction examples created successfully")
content = content.replace("label.x = 0.15", "label.x = 0.1")

# Ensure time is in the data.frame
df_old = """  full_df_loc <- data.frame(
    measured = test_loc$predicted + test_loc$residual,
    base     = test_loc$predicted,
    rf       = test_loc$predicted + zs_preds
  )"""
df_new = """  full_df_loc <- data.frame(
    time     = test_loc$time,
    measured = test_loc$predicted + test_loc$residual,
    base     = test_loc$predicted,
    rf       = test_loc$predicted + zs_preds
  )
  full_df_loc <- full_df_loc[order(full_df_loc$time), ]"""
content = content.replace(df_old, df_new)

# Replace make_residual_hist with make_pred_plot
old_plot = "  excerpt_panels[[target]] <- make_residual_hist(full_df_loc, title_str, has_lstm = FALSE)"
new_plot = "  excerpt_panels[[target]] <- make_pred_plot(head(full_df_loc, 120), title_str, has_lstm = FALSE, show_legend = TRUE)"
content = content.replace(old_plot, new_plot)

with open(out_file, "w") as f:
    f.write(content)

print(f"Wrote {out_file}")
