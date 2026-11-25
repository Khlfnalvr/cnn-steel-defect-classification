#!/usr/bin/env bash
# Unduh NEU Surface Defect Database (1800 citra .bmp) ke data/NEU
# Mirror GitHub dari dataset asli Northeastern University (Song & Yan, 2013).
set -e
mkdir -p data
git clone --depth 1 https://github.com/TheAyushB/Manufacturing-Defects.git /tmp/neu_mirror
rm -rf data/NEU && cp -r "/tmp/neu_mirror/NEU Metal Surface Defects Data" data/NEU
echo "Jumlah citra: $(find data/NEU -name '*.bmp' | wc -l)"
