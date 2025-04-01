#!/bin/bash
echo "testing generic functionality"
cd tests_generic/ || exit
python3 -m pytest -v
cd .. || exit

echo "testing lab specific metrics"
cd tests_labs/ || exit
python3 -m pytest -v
cd .. || exit
