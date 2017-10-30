#!/bin/sh


cd /home/jack/Dropbox/dpca/new/cued/gaus30_bin10/0059/multi_d/
pwd
python /home/jack/classification_scripts/dpca_setup_multirp.py
python /home/jack/classification_scripts/analyze_dpca.py
Rscript ~/classification_scripts/dpca_plot_analysis.R

cd /home/jack/Dropbox/dpca/new/cued/gaus30_bin10/504/multi_nod/
pwd
python /home/jack/classification_scripts/dpca_setup_multirp_nod.py
python /home/jack/classification_scripts/analyze_dpca_nod.py
Rscript ~/classification_scripts/dpca_plot_analysisz_nod.R

cd /home/jack/Dropbox/dpca/new/cued/gaus30_bin10/504/multi_d/
pwd
python /home/jack/classification_scripts/dpca_setup_multirp.py
python /home/jack/classification_scripts/analyze_dpca.py
Rscript ~/classification_scripts/dpca_plot_analysis.R


"""
cd /home/jack/Dropbox/dpca/new/uncued/gaus30_bin10/0059/nod/
pwd
python /home/jack/classification_scripts/dpca_setup_binaryrp_nod.py
python /home/jack/classification_scripts/analyze_dpca_nod.py

cd /home/jack/Dropbox/dpca/new/uncued/gaus30_bin10/504/nod/
pwd
python /home/jack/classification_scripts/dpca_setup_binaryrp_nod.py
python /home/jack/classification_scripts/analyze_dpca_nod.py

cd /home/jack/Dropbox/dpca/new/cued/gaus30_bin10/0059/test_nod/
pwd
python /home/jack/classification_scripts/dpca_setup_binaryrp_nod.py
python /home/jack/classification_scripts/analyze_dpca_nod.py

cd /home/jack/Dropbox/dpca/new/cued/gaus30_bin10/504/test_nod/
pwd
python /home/jack/classification_scripts/dpca_setup_binaryrp_nod.py
python /home/jack/classification_scripts/analyze_dpca_nod.py


cd /home/jack/Dropbox/dpca/new/uncued/gaus30_bin10/504/test_100shuff/
pwd
python /home/jack/classification_scripts/dpca_setup_binaryrp.py
python /home/jack/classification_scripts/analyze_dpca.py

cd /home/jack/Dropbox/dpca/new/cued/gaus30_bin10/0059/100_shuff/
pwd
python /home/jack/classification_scripts/dpca_setup_binaryrp.py
python /home/jack/classification_scripts/analyze_dpca.py

cd /home/jack/Dropbox/dpca/new/cued/gaus30_bin10/504/100_shuff/
pwd
python /home/jack/classification_scripts/dpca_setup_binaryrp.py
python /home/jack/classification_scripts/analyze_dpca.py

"""

exec bash


