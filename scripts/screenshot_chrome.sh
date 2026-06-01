#!/bin/bash
# Use Chrome headless to screenshot the full page at high resolution,
# then use Python/PIL to crop sections by their pixel positions.

set -e

OUTDIR="/tmp/docx_screenshots"
HTML="file:///home/ubuntu/Projects/大模型提取特征/outputs/system_architecture_v4.html"
FULLPAGE="$OUTDIR/fullpage.png"

echo "Taking full page screenshot..."
google-chrome \
  --headless=new \
  --disable-gpu \
  --no-sandbox \
  --disable-dev-shm-usage \
  --window-size=1100,20000 \
  --screenshot="$FULLPAGE" \
  --hide-scrollbars \
  --force-device-scale-factor=2 \
  "$HTML" 2>/dev/null

echo "Full page screenshot: $(ls -lh $FULLPAGE | awk '{print $5}')"
echo "Done with Chrome"
