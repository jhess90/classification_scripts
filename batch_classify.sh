#!/bin/sh

batch_classify_path="~/documents/lab/workspace/Classification_scripts"

if [ $# -eq 0 ]; then
    echo "No files provided"
    exit 1

fi

#chmood +x ~/documents/lab/workspace/Classifier_scripts/jack_classifier.py

python jack_classifier.py
