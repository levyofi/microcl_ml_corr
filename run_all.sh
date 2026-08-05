export UV_OFFLINE=1
export KERAS_HOME=$PWD
for script in inst/examples/*/run_scenario_*.R; do
  echo "Running $script"
  Rscript "$script"
done
