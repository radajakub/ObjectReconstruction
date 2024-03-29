# ObjectReconstruction
3D Object reconstruction from photos as a semestral project for Three-Dimensional Computer Vision

The goal is to reconstruct a 3D point cloud representing and object from a given set of photos of the object from multiple views using methods from 3D computer vision.

## Example
Construct a 3D point cloud from this set of 12 photos capturing a gate of a church in Prague.
![all_images](https://github.com/radajakub/ObjectReconstruction/assets/101521391/32e9513c-40e1-4d1f-9747-cf5e7f14e8ce)

The resulting point cloud looks like this.
<img width="1049" alt="cloud" src="https://github.com/radajakub/ObjectReconstruction/assets/101521391/bccaf84c-f79d-4c95-9b9c-913e854e2155">


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
Without specifing this option, the logs are printed to stdout and the image is shown.

## Show all images of a scene

Executing `python src/all_images.py --scene <scene_name>` will show all images in the folder `<scene_name>` displayed in a grid.
E.g. `python src/all_images.py --scene scene_1`.

If `--out <path>` is specified, the resulting image is saved into folder given by `<path>`.
Without specifing this option, image is shown.

## Epipolar geometry

Script for executing epipolar geometry between two images is located in `src/epipolar_geometry.py`.

It has multiple parameters tuning the behaviour of the algorithm:

**Required**
- `--scene <scene_name>` - name of the scene folder in `data/` folder
- `--image1 <image_number>` - number of the first image
- `--image2 <image_number>` - number of the second image
**Optional**
- `--out <path>` - path to the output folder where all outputs are saved (if not provided, the outputs are only shown)
- `--seed <int>` - seed for the random generator (defuault from system)
- `--threshold <float>` - threshold for the RANSAC algorithm (`default 3`)
- `--max_iter <int>` - maximum number of iterations for the RANSAC algorithm (`default 1000`) in case the computed condition fails
- `--p <float>` - probability for computing the stop condition for the RANSAC algorithm (`default 0.9999`)
- `--silent <bool>` - if `True`, the logs are not shown (`default False`)

For example: `python src/epipolar_geometry.py --scene scene_1 --img1 1 --img2 2` show epipolar geometry estimate between images `1` and `2` in scene `scene_1`.

## Sparse point cloud

Script to compute sparse point cloud from a scene and initial pair of cameras is located in `src/sparse.py`. It has multiple parameters tuning the behaviour of the algorithm:

**Required**
- `--scene <scene_name>` - name of the scene folder in `data/` folder
- `--image1 <image_number>` - number of the first intial image
- `--image2 <image_number>` - number of the second initial image
**Optional**
- `--out <path>` - path to the output folder where all outputs are saved (if not provided, the outputs are only shown)
- `--seed <int>` - seed for the random generator (defuault from system)
- `--threshold <float>` - threshold for the RANSAC algorithm (`default 3`)
- `--max_iter <int>` - maximum number of iterations for the RANSAC algorithm (`default 1000`) in case the computed condition fails
- `--p <float>` - probability for computing the stop condition for the RANSAC algorithm (`default 0.9999`)
- `--silent <bool>` - if `True`, the logs are not shown (`default False`)
- `--pose-threshold <float>` - threshold for the RANSAC algorithm for gluing the camera to the already glued set (`default 3)
- `--reprojection-threshold <float>` - threshold for selecting inliers for newly appended camera (`default 3`)

For example: `python src/sparse.py --scene scene_1 --img1 5 --img2 6 --out out` computes the sparse point cloud for scene `scene_1` using images `5` and `6` as initial pair of cameras and saves the point cloud and the cameras into `out/scene_1` folder.


## Dense point cloud

This step is divided into two parts because we need to compute disparity maps in Matlab.

### Rectifying images and computing disparity maps

Script is located in `src/dense_first.py`. It loads sparse point cloud and cameras from previous step, computes rectified images for preselected camera pairs and prepares tasks to compute disparity maps.

It has multiple parameters tuning the behaviour of the algorithm:

**Required**
- `--scene <scene_name>` - name of the scene folder in `data/` folder
- `--in <path>` - path to the input folder where the sparse point cloud is saved (output of previous task)
**Optional**
- `--out <path>` - path to the output folder where all outputs are saved (if not provided, the outputs are only shown)
- `--seed <int>` - seed for the random generator (defuault from system)
- `--silent <bool>` - if `True`, the logs are not shown (`default False`)
- `--fundamental-threshold <float>` - threshold for computing inliers from disparity maps using sampson reprojection error (`default 0.05`)

The rectified images and data for computing disparity maps in Matlab are saved into `<out>/<scene>/rectified/stereo_in.mat`.

### Compute disparity maps

Use `stereo_in.mat` in Matlab script to create `stereo_out.mat` with disparity maps. Copy it to the same folder used for `stereo_in.mat`.

### Computing dense point cloud

Script to compute dense point cloud from sparse point cloud and disparities computed by previous scripts is in `src/dense_second.py`. It saves the dense point and shows the disparity maps computed by Matlab algorithm.

It has multiple parameters tuning the behaviour of the algorithm:

**Required**
- `--scene <scene_name>` - name of the scene folder in `data/` folder
- `--in <path>` - path to the input folder where the sparse point cloud is saved, together with the rectified images and disparity maps (output of previous task and the matlab script)
**Optional**
- `--out <path>` - path to the output folder where all outputs are saved (if not provided, the outputs are only shown)
- `--seed <int>` - seed for the random generator (defuault from system)
- `--silent <bool>` - if `True`, the logs are not shown (`default False`)
- `--fundamental-threshold <float>` - threshold for computing inliers from disparity maps using sampson reprojection error (`default 0.05`)

For example: `python src/dense_second.py --scene scene_1 --in dense_first --out dense_second` computes the dense point cloud for scene `scene_1` and saves it together with disparity maps.
The disparity maps from the intermediate step are loaded from `<out>/<scene>/rectified/stereo_out.mat`.

