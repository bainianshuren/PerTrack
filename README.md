\# PerTrack: A High-Precision Pedestrian Tracking Algorithm Based on Multi-Feature Fusion and Trajectory Optimization



\## Project Description

This project implements a high-precision pedestrian tracking model PerTrack, which is optimized based on the DeepSORT framework. The model improves the tracking accuracy of pedestrians in complex scenarios (such as occlusion, fast movement, and camera shake) by designing a Multi-Feature Fusion Module (MFFM), a Dynamic Trajectory Smoothing Module (DTSM), and an Occlusion-Aware Re-identification Network (OAR-Net). 



The project includes complete code for model training, testing, ablation experiments, and public dataset preprocessing, which can directly reproduce all experimental results in the paper. The core innovations of PerTrack are reflected in adaptive feature representation, robust trajectory correction, and occlusion-aware re-identification, achieving a good balance between tracking precision, recall and real-time performance, and having potential application value in intelligent surveillance, autonomous driving and other fields.



\## Dataset Information

\### Supported Datasets

1\. MOT17 Dataset

&#x20;  - Source: Multiple Object Tracking Benchmark 2017, including video sequences from different scenes (street, square, campus)

&#x20;  - Official Download: https://motchallenge.net/data/MOT17/

&#x20;  - Preprocessing Specification: Frame extraction from video sequences (25fps), image resolution unified to 1920×1080, pixel value normalized to \[0,1]; annotation files converted to the format of (frame\_id, pedestrian\_id, xmin, ymin, xmax, ymax, confidence, class, visibility). A total of 14 training sequences and 14 test sequences are included.



2\. MOT20 Dataset

&#x20;  - Source: Multiple Object Tracking Benchmark 2020, focusing on dense pedestrian scenarios

&#x20;  - Official Download: https://motchallenge.net/data/MOT20/

&#x20;  - Preprocessing Specification: Consistent with MOT17 preprocessing rules; the dataset includes 8 training sequences and 8 test sequences, with higher pedestrian density and more severe occlusion situations.



\### Data Annotation Format

Pedestrian annotation adopts the standard MOT challenge format `(frame\_id, pedestrian\_id, xmin, ymin, xmax, ymax, confidence, class, visibility)`. The project provides a label conversion script to adapt to the input requirements of the PerTrack model, supporting mutual conversion between original MOT annotation format and the format required by the model.



\## Code Information

\### Core Framework

Python + PyTorch (modular design, low coupling between modules, easy to modify and expand; supports mixed precision training to accelerate training process)



\### Code Structure

PerTrack/

├── models/        Core code of PerTrack, including MFFM, DTSM, OAR-Net and overall tracking network construction

├── data/          Dataset preprocessing, loading and augmentation scripts (supports MOT17/MOT20)

├── train/         Model training script (learning rate scheduling, optimizer configuration, loss function definition)

├── test/          Model testing and metric calculation script (MOTA, IDF1, IDP, IDR, FPS, etc.)

├── ablation/      Ablation experiment code, verify the effectiveness of each module independently

├── utils/         Tool functions (logging, weight saving, visualization, trajectory post-processing, metric calculation)

├── weights/       Pre-trained optimal weight file of PerTrack (can be directly used for testing and inference)

├── README.md      Project description and usage instructions

└── requirements.txt Dependent library list



\### Compatibility

\- Support Windows/Linux operating system

\- Compatible with Python 3.8 and above

\- Support GPU training/inference (CUDA 11.7+ recommended) and CPU inference



\## Usage Methods

\### 1. Environment Preparation

Install all dependent libraries with one click through the `requirements.txt` file in the root directory:

```bash

pip install -r requirements.txt

```



\### 2. Dataset Preparation

1\. Download the original MOT17 and MOT20 datasets from the official links above;

2\. Run the preprocessing script to complete frame extraction, format conversion and normalization:

```bash

\# Preprocess MOT17 dataset

python data/preprocess.py --dataset MOT17 --raw\_path \[Your raw MOT17 path] --save\_path \[Your save path]



\# Preprocess MOT20 dataset

python data/preprocess.py --dataset MOT20 --raw\_path \[Your raw MOT20 path] --save\_path \[Your save path]

```



\### 3. Model Training

Run the training script, support custom dataset, batch size, training epochs and other parameters:

```bash

python train/train.py --dataset MOT17 --data\_path \[Your preprocessed data path] --batch\_size 16 --epochs 80 --lr 2e-4

```

\- Key training parameters: initial learning rate 2e-4 (adjusted to 2e-5 after 30 epochs, 2e-6 after 60 epochs), AdamW optimizer, combined loss of triplet loss + cross entropy loss + smooth L1 loss, batch size 16, total training epochs 80.



\### 4. Model Testing

Load the pre-trained optimal weight file, run the test script, and automatically calculate and output evaluation metrics:

```bash

python test/test.py --dataset MOT17 --data\_path \[Your preprocessed data path] --weight\_path weights/best\_PerTrack.pth

```

\- The script outputs MOTA, MOTP, IDF1, IDP, IDR, FPS, Params, GFLOPs and other metrics by default, consistent with the experimental settings in the paper.



\### 5. Ablation Experiment

Verify the effectiveness of a single module (MFFM/DTSM/OAR-Net) or combined modules independently:

```bash

\# Verify the combined effect of MFFM and DTSM modules

python ablation/ablation.py --dataset MOT17 --data\_path \[Your preprocessed data path] --modules MFFM DTSM

```

\- The ablation experiment is based on the DeepSORT baseline model, and 7 groups of experimental configurations are supported (3 single modules, 3 double-module combinations, 1 three-module combination).



\### 6. Model Inference

Support inference and visualization of single video/file of video frames, and save the tracking results (bounding box + pedestrian ID + confidence):

```bash

python test/infer.py --video\_path \[Your single video path] --weight\_path weights/best\_PerTrack.pth --save\_path \[Your inference result save path]

```

\- Support saving tracking results in video format (with bounding box and ID drawing) and txt format (consistent with MOT challenge submission format).



\## Dependencies (Python Libraries)

| Library Name      | Version   | Function Description                                                                 |

|-------------------|-----------|--------------------------------------------------------------------------------------|

| Python            | 3.8.18    | Basic development environment                                                        |

| PyTorch           | 1.14.0    | Deep learning framework (model building/training)                                    |

| Torchvision       | 0.15.1    | Computer vision toolkit (data augmentation, feature extraction)                      |

| NumPy             | 1.24.3    | Numerical calculation and array processing                                            |

| Pandas            | 1.5.3     | Data processing and annotation file parsing                                          |

| OpenCV-Python     | 4.8.0     | Video/frame reading, preprocessing and visualization                                  |

| Matplotlib        | 3.7.1     | Experimental result visualization and curve plotting                                  |

| Scikit-learn      | 1.2.2     | Metric calculation and feature clustering                                            |

| tqdm              | 4.65.0    | Training process progress bar                                                        |

| Pillow            | 9.5.0     | Image format processing                                                              |

| scipy             | 1.10.1    | Scientific computing (trajectory smoothing, distance calculation)                     |

| filterpy          | 1.4.5     | Kalman filter implementation (trajectory prediction)                                 |



All dependent libraries and their specified versions are listed in `requirements.txt` to ensure environment consistency and avoid version compatibility issues.



\## Data Processing / Modeling Steps

\### Data Processing Steps

1\. Raw Data Acquisition: Download the original video sequences and corresponding annotation files of MOT17/MOT20 datasets from the official channel;

2\. Format Conversion: Extract frames from video sequences (25fps) and convert to PNG format; convert original annotation files to the txt format adapted to PerTrack model;

3\. Image Preprocessing:

&#x20;  - Resolution Unification: Resize all frames to 1920×1080 resolution to ensure input consistency;

&#x20;  - Normalization: Normalize pixel values of frames to \[0,1] to accelerate model convergence;

&#x20;  - Data Augmentation: Adopt random flipping, brightness/contrast adjustment, Gaussian blur, random cropping and other strategies in the training stage to improve the generalization ability of the model;

4\. Dataset Division: Use the official division of MOT challenge (fixed training/test sequences) to ensure experiment reproducibility;

5\. Data Loading: Build a data loader using PyTorch's DataLoader with a batch size of 16 and support multi-threaded loading (num\_workers=8 recommended).



\### Modeling \& Training Steps

1\. Model Construction: Based on the DeepSORT baseline framework, replace the feature extraction network with MFFM, add DTSM to the trajectory prediction module, and integrate OAR-Net into the re-identification branch to build the overall PerTrack network;

2\. Hyperparameter Setting: Consistent with the paper, initial learning rate 2e-4 (adjusted to 2e-5 after 30 epochs, 2e-6 after 60 epochs), total 80 epochs, AdamW optimizer, combined loss of triplet loss + cross entropy loss + smooth L1 loss;

3\. Model Training: End-to-end training on the training set, with early stopping strategy (stop training and save the optimal weight when the validation set MOTA does not improve for 8 consecutive epochs);

4\. Model Testing: Load the optimal weight on the test set for tracking inference, calculate MOTA, MOTP, IDF1, IDP, IDR (standard MOT challenge metrics), and count Params, FPS, GFLOPs;

5\. Ablation Experiment: Based on DeepSORT, add MFFM/DTSM/OAR-Net modules separately and in combination to verify the effectiveness and synergy of each module;

6\. Result Visualization: Visualize the tracking results (draw bounding boxes, pedestrian IDs and confidence), and compare the tracking effects of different models under complex scenarios (occlusion, fast movement).



\## Citation

If the code or model of this project is used in related research, please cite the original paper:

\[To be filled] PerTrack: A High-Precision Pedestrian Tracking Algorithm Based on Multi-Feature Fusion and Trajectory Optimization.



\## License

This project is released under the MIT License, which permits free use, copying, modification, distribution, and secondary development for both commercial and non-commercial purposes, with the only requirement to retain the original author's copyright notice and license statement.



The full license text is available in the `LICENSE` file in the root directory of the repository, and the official description can be viewed at: https://opensource.org/licenses/MIT



\## Reproducibility Guarantee

1\. Fixed Random Seed: Set the random seeds of Python, NumPy and PyTorch in all codes to ensure the consistency of random operations in the training process;

2\. Pre-trained Optimal Weights: The repository provides the pre-trained optimal weight file of PerTrack, which can directly run the test script without re-training to reproduce all experimental metrics in the paper;

3\. Unified Experimental Environment: Clearly specify the version of all dependent libraries and provide a one-click installation script to avoid result deviations caused by environmental differences;

4\. Complete Experimental Code: Include all codes for training, testing and ablation experiments without hidden logic, and all parameter settings are completely consistent with the paper;

5\. Standardized Preprocessing: Provide the same dataset preprocessing script as the paper, and the input original public dataset can obtain the preprocessed data consistent with the experiment;

6\. Standard Metric Calculation: Adopt the official MOT challenge metric calculation code to ensure the consistency of evaluation results with the paper.

