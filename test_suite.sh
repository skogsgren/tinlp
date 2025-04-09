#!/bin/bash
cd tests_generic/ || exit
python3 -m pytest -v
cd .. || exit

cd tests_labs/ || exit
python3 -m pytest -v
cd .. || exit
