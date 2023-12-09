#!/bin/bash

CAMERAS=(1 2 3 4 5 6 7 8 9 10 11 12)

for i in ${CAMERAS[*]}; do
    for j in ${CAMERAS[*]}; do
        if [[ $i -ne $j ]]; then
            echo "$i $j"
            python src/task1.py --scene scene_1 --img1 $i --img2 $j --out results/epipolar/$i\_$j/
        fi
    done
done
