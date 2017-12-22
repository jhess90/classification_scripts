#!/bin/sh

cd /home/jack/Dropbox/corr_analysis/corr_temp/

cd /home/jack/Dropbox/corr_analysis/corr_temp/0_3_10_1/
mkdir z_only g_only both_10 
rm *.npy *.xlsx *.png M1* S1* PmD* master*
mv Extracted* z_only/
cd z_only/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_zscoreonly.py
rm *.npy
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../g_only/
cd ../g_only/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_smoothonly.py
rm *.npy
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../both_10/
cd ../both_10/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_both_10.py
rm *.npy
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../
cd ../

cd ../0_3_10_2/
mkdir z_only g_only both_10 
rm *.npy *.xlsx *.png M1* S1* PmD* master*
mv Extracted* z_only/
cd z_only/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_zscoreonly.py
rm *.npy
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../g_only/
cd ../g_only/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_smoothonly.py
rm *.npy
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../both_10/
cd ../both_10/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_both_10.py
rm *.npy
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../
cd ../

cd ../0_3_10_3/
mkdir z_only g_only both_10 
rm *.npy *.xlsx *.png M1* S1* PmD* master*
mv Extracted* z_only/
cd z_only/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_zscoreonly.py
rm *.npy
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../g_only/
cd ../g_only/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_smoothonly.py
rm *.npy
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../both_10/
cd ../both_10/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_both_10.py
rm *.npy
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../
cd ../

cd ../0_3_13_1/
mkdir z_only g_only both_10 
rm *.npy *.xlsx *.png M1* S1* PmD* master*
mv Extracted* z_only/
cd z_only/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_zscoreonly.py
rm *.npy
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../g_only/
cd ../g_only/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_smoothonly.py
rm *.npy
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../both_10/
cd ../both_10/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_both_10.py
rm *.npy
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../
cd ../

cd ../0_3_13_2/
mkdir z_only g_only both_10 
rm *.npy *.xlsx *.png M1* S1* PmD* master*
mv Extracted* z_only/
cd z_only/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_zscoreonly.py
rm *.npy
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../g_only/
cd ../g_only/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_smoothonly.py
rm *.npy
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../both_10/
cd ../both_10/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_both_10.py
rm *.npy
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../
cd ../

cd ../0_3_13_3/
mkdir z_only g_only both_10 
rm *.npy *.xlsx *.png M1* S1* PmD* master*
mv Extracted* z_only/
cd z_only/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_zscoreonly.py
rm *.npy
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../g_only/
cd ../g_only/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_smoothonly.py
rm *.npy
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../both_10/
cd ../both_10/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_both_10.py
rm *.npy
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../
cd ../

cd ../0_3_14_1/
mkdir z_only g_only both_10 
rm *.npy *.xlsx *.png M1* S1* PmD* master*
mv Extracted* z_only/
cd z_only/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_zscoreonly.py
rm *.npy
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../g_only/
cd ../g_only/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_smoothonly.py
rm *.npy
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../both_10/
cd ../both_10/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_both_10.py
rm *.npy
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../
cd ../

cd ../0_3_14_2/
mkdir z_only g_only both_10 
rm *.npy *.xlsx *.png M1* S1* PmD* master*
mv Extracted* z_only/
cd z_only/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_zscoreonly.py
rm *.npy
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../g_only/
cd ../g_only/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_smoothonly.py
rm *.npy
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../both_10/
cd ../both_10/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_both_10.py
rm *.npy
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../
cd ../

cd ../0_3_14_3/
mkdir z_only g_only both_10 
rm *.npy *.xlsx *.png M1* S1* PmD* master*
mv Extracted* z_only/
cd z_only/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_zscoreonly.py
rm *.npy
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../g_only/
cd ../g_only/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_smoothonly.py
rm *.npy
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../both_10/
cd ../both_10/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_both_10.py
rm *.npy
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../
cd ../

cd ../0_3_27_1/
mkdir z_only g_only both_10 
rm *.npy *.xlsx *.png M1* S1* PmD* master*
mv Extracted* z_only/
cd z_only/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_zscoreonly.py
rm *.npy
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../g_only/
cd ../g_only/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_smoothonly.py
rm *.npy
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../both_10/
cd ../both_10/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_both_10.py
rm *.npy
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../
cd ../

cd ../0_3_27_2/
mkdir z_only g_only both_10 
rm *.npy *.xlsx *.png M1* S1* PmD* master*
mv Extracted* z_only/
cd z_only/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_zscoreonly.py
rm *.npy
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../g_only/
cd ../g_only/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_smoothonly.py
rm *.npy
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../both_10/
cd ../both_10/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_both_10.py
rm *.npy
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../
cd ../

cd ../0_3_28_1/
mkdir z_only g_only both_10 
rm *.npy *.xlsx *.png M1* S1* PmD* master*
mv Extracted* z_only/
cd z_only/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_zscoreonly.py
rm *.npy
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../g_only/
cd ../g_only/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_smoothonly.py
rm *.npy
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../both_10/
cd ../both_10/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_both_10.py
rm *.npy
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../
cd ../

cd ../0_3_28_2/
mkdir z_only g_only both_10 
rm *.npy *.xlsx *.png M1* S1* PmD* master*
mv Extracted* z_only/
cd z_only/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_zscoreonly.py
rm *.npy
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../g_only/
cd ../g_only/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_smoothonly.py
rm *.npy
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../both_10/
cd ../both_10/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_both_10.py
rm *.npy
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../
cd ../

cd ../0_3_28_3/
mkdir z_only g_only both_10 
rm *.npy *.xlsx *.png M1* S1* PmD* master*
mv Extracted* z_only/
cd z_only/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_zscoreonly.py
rm *.npy
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../g_only/
cd ../g_only/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_smoothonly.py
rm *.npy
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../both_10/
cd ../both_10/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_both_10.py
rm *.npy
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../
cd ../

cd ../0_3_10_2/
mkdir z_only g_only both_10 
rm *.npy *.xlsx *.png M1* S1* PmD* master*
mv Extracted* z_only/
cd z_only/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_zscoreonly.py
rm *.npy
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../g_only/
cd ../g_only/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_smoothonly.py
rm *.npy
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../both_10/
cd ../both_10/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_both_10.py
rm *.npy
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../
cd ../

cd ../0_3_9_1/
mkdir z_only g_only both_10 
rm *.npy *.xlsx *.png M1* S1* PmD* master*
mv Extracted* z_only/
cd z_only/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_zscoreonly.py
rm *.npy
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../g_only/
cd ../g_only/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_smoothonly.py
rm *.npy
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../both_10/
cd ../both_10/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_both_10.py
rm *.npy
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../
cd ../

cd ../0_3_9_2/
mkdir z_only g_only both_10 
rm *.npy *.xlsx *.png M1* S1* PmD* master*
mv Extracted* z_only/
cd z_only/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_zscoreonly.py
rm *.npy
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../g_only/
cd ../g_only/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_smoothonly.py
rm *.npy
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../both_10/
cd ../both_10/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_both_10.py
rm *.npy
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../
cd ../

cd ../0_3_9_3/
mkdir z_only g_only both_10 
rm *.npy *.xlsx *.png M1* S1* PmD* master*
mv Extracted* z_only/
cd z_only/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_zscoreonly.py
rm *.npy
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../g_only/
cd ../g_only/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_smoothonly.py
rm *.npy
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../both_10/
cd ../both_10/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_both_10.py
rm *.npy
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../
cd ../

cd ../5_3_10_1/
mkdir z_only g_only both_10 
rm *.npy *.xlsx *.png M1* S1* PmD* master*
mv Extracted* z_only/
cd z_only/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_zscoreonly.py
rm *.npy
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../g_only/
cd ../g_only/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_smoothonly.py
rm *.npy
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../both_10/
cd ../both_10/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_both_10.py
rm *.npy
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../
cd ../

cd ../5_3_10_2/
mkdir z_only g_only both_10 
rm *.npy *.xlsx *.png M1* S1* PmD* master*
mv Extracted* z_only/
cd z_only/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_zscoreonly.py
rm *.npy
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../g_only/
cd ../g_only/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_smoothonly.py
rm *.npy
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../both_10/
cd ../both_10/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_both_10.py
rm *.npy
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../
cd ../

cd ../5_3_10_3/
mkdir z_only g_only both_10 
rm *.npy *.xlsx *.png M1* S1* PmD* master*
mv Extracted* z_only/
cd z_only/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_zscoreonly.py
rm *.npy
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../g_only/
cd ../g_only/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_smoothonly.py
rm *.npy
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../both_10/
cd ../both_10/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_both_10.py
rm *.npy
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../
cd ../

cd ../5_3_13_1/
mkdir z_only g_only both_10 
rm *.npy *.xlsx *.png M1* S1* PmD* master*
mv Extracted* z_only/
cd z_only/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_zscoreonly.py
rm *.npy
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../g_only/
cd ../g_only/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_smoothonly.py
rm *.npy
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../both_10/
cd ../both_10/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_both_10.py
rm *.npy
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../
cd ../

cd ../5_3_13_2/
mkdir z_only g_only both_10 
rm *.npy *.xlsx *.png M1* S1* PmD* master*
mv Extracted* z_only/
cd z_only/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_zscoreonly.py
rm *.npy
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../g_only/
cd ../g_only/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_smoothonly.py
rm *.npy
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../both_10/
cd ../both_10/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_both_10.py
rm *.npy
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../
cd ../

cd ../5_3_13_3/
mkdir z_only g_only both_10 
rm *.npy *.xlsx *.png M1* S1* PmD* master*
mv Extracted* z_only/
cd z_only/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_zscoreonly.py
rm *.npy
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../g_only/
cd ../g_only/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_smoothonly.py
rm *.npy
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../both_10/
cd ../both_10/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_both_10.py
rm *.npy
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../
cd ../

cd ../5_3_14_1/
mkdir z_only g_only both_10 
rm *.npy *.xlsx *.png M1* S1* PmD* master*
mv Extracted* z_only/
cd z_only/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_zscoreonly.py
rm *.npy
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../g_only/
cd ../g_only/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_smoothonly.py
rm *.npy
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../both_10/
cd ../both_10/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_both_10.py
rm *.npy
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../
cd ../

cd ../5_3_14_2/
mkdir z_only g_only both_10 
rm *.npy *.xlsx *.png M1* S1* PmD* master*
mv Extracted* z_only/
cd z_only/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_zscoreonly.py
rm *.npy
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../g_only/
cd ../g_only/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_smoothonly.py
rm *.npy
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../both_10/
cd ../both_10/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_both_10.py
rm *.npy
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../
cd ../

cd ../5_3_14_3/
mkdir z_only g_only both_10 
rm *.npy *.xlsx *.png M1* S1* PmD* master*
mv Extracted* z_only/
cd z_only/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_zscoreonly.py
rm *.npy
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../g_only/
cd ../g_only/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_smoothonly.py
rm *.npy
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../both_10/
cd ../both_10/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_both_10.py
rm *.npy
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../
cd ../

cd ../5_3_28_2/
mkdir z_only g_only both_10 
rm *.npy *.xlsx *.png M1* S1* PmD* master*
mv Extracted* z_only/
cd z_only/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_zscoreonly.py
rm *.npy
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../g_only/
cd ../g_only/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_smoothonly.py
rm *.npy
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../both_10/
cd ../both_10/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_both_10.py
rm *.npy
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../
cd ../

cd ../5_3_28_3/
mkdir z_only g_only both_10 
rm *.npy *.xlsx *.png M1* S1* PmD* master*
mv Extracted* z_only/
cd z_only/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_zscoreonly.py
rm *.npy
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../g_only/
cd ../g_only/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_smoothonly.py
rm *.npy
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../both_10/
cd ../both_10/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_both_10.py
rm *.npy
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../
cd ../

cd ../5_3_9_1/
mkdir z_only g_only both_10 
rm *.npy *.xlsx *.png M1* S1* PmD* master*
mv Extracted* z_only/
cd z_only/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_zscoreonly.py
rm *.npy
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../g_only/
cd ../g_only/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_smoothonly.py
rm *.npy
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../both_10/
cd ../both_10/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_both_10.py
rm *.npy
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../
cd ../

cd ../5_3_9_2/
mkdir z_only g_only both_10 
rm *.npy *.xlsx *.png M1* S1* PmD* master*
mv Extracted* z_only/
cd z_only/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_zscoreonly.py
rm *.npy
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../g_only/
cd ../g_only/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_smoothonly.py
rm *.npy
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../both_10/
cd ../both_10/
pwd
python /home/jack/workspace/classification_scripts/corr_analysis_both_10.py
rm *.npy
Rscript /home/jack/workspace/classification_scripts/corr_plotting.R
mv Extracted* ../
cd ../


exec bash
