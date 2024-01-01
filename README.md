# ObjectReconstruction
3D Object reconstruction from photos as a semestral project for Three-Dimensional Computer Vision

## Installation

Recommended version of Python is `3.11.5`.
Install packages in `requirements.txt` with `pip install -r requirements.txt`.

Compile p5 algorithm for your platform:
    - `cd p5/src-python`
    - `make`
    - copy files `ext.py` and the `.so` file from `python` folder to `src/p5/` folder

## RANSAC task

Executing `python src/task_ransac.py --scene data/ransac` will run example RANSAC algorithm for fitting points with a line.
The data points are in `data/ransac/ransac.txt` but the generating line is hard-coded in the script as it is only an example.

If `--out <path>` is specified, the resulting image and log entries are saved into folder given by `<path>`.
