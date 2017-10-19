#!/bin/sh
python /home/jack/workspace/classification_scripts/dpca_setup_binaryrp.py
python /home/jack/workspace/classification_scripts/analyze_dpca.py
Rscript /home/jack/workspace/classification_scripts/dpca_plot_analysis.R
