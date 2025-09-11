#!/bin/bash
set -e

echo "=== Starting dataset download ==="
python download_dataset.py

echo "=== Running preprocessing, mixing, and sharding ==="
python main.py --debug True

echo "=== Pipeline completed successfully! ==="