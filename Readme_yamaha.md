# Lead Instrument Detection
[[github repo](https://github.com/Sonata165/SoloDetection)]

Author: [Longshen Ou](http://oulongshen.xyz/), 2024/7/25

This is a repository of Longshen's internship project at YAMAHA (2024/05 ~ 2024/08), about detecting lead instruments from multitrack audio.

- [Lead Instrument Detection](#lead-instrument-detection)
  - [Directories](#directories)
    - [Code directory](#code-directory)
    - [Data directory](#data-directory)
      - [Local data dir](#local-data-dir)
      - [Remote data dir](#remote-data-dir)
  - [Prepare environment](#prepare-environment)
    - [MacBook](#macbook)
    - [Linux Server](#linux-server)
  - [Data Preprocessing](#data-preprocessing)
    - [Data Annotation](#data-annotation)
      - [MJN](#mjn)
      - [MedleyDB v1](#medleydb-v1)
    - [Constructing segment-level dataset](#constructing-segment-level-dataset)
  - [Training and testing](#training-and-testing)
  - [Visualize results](#visualize-results)
  - [Where can the performance be enhanced](#where-can-the-performance-be-enhanced)



## Directories

There are **3 relevant directories in total**. One for code (`/Users/longshen/Code/SoloDetection`), one for data annotation (on laptop, `/Users/longshen/Code/Datasets`), the other for model training and testing (on server, `/home/longshen/data`).

There is an copy of code directory on the server for running the training and testing, located at `/home/longshen/work/SoloDetection`.

### Code directory
The location where you clone this repository is the code directory. The structure is as follows:
    
    .
    ├── data_preprocessing              Scripts for creating metadata and segment-level datasets
        ├── copy_mixture_audio.py           Scripts that copy mixture audio to a separate folder 
                                                to facilitate SVM feature extraction
        ├── dataset_statistics.py           Obtain dataset statistics
        ├── segment_dataset_medley.py       Create segment-level dataset for MedleyDB dataset
        └── segment_dataset_mjn.py          Create segment-level dataset for MJN dataset
    ├── hparams                         Hyperparameters/configurations for all solo detection models
        ├── ch_cls                          Channel classification models (train and test with MJN dataset)
            ├── ch_permute.yaml                 Our final model
            ├── freeze_mert.yaml                Ablation: freeze MERT during training
            ├── no_ch_emb.yaml                  Ablation: remove channel embedding
            ├── no_inst_emb.yaml                Ablation: remove instrument embedding
            ├── no_oracle_mix.yaml              Ablation: remove mixture channel from human
            ├── no_permute.yaml                 Ablation: remove channel permutation
            └── train_last_layer.yaml           Ablation: only train MERT's last Transformer layer
        ├── crnn                            CRNN baselines (train and test with MJN dataset)
            ├── crnn_mert_attn_ft.yaml          CRNN + MERT + channel attention + fine tuning
            ├── crnn_mert_ft.yaml               CRNN + MERT + fine tuning
            ├── crnn_mert.yaml                  CRNN + MERT feature
            └── crnn.yaml                       CRNN + mel spectrogram as input
        ├── cross_dataset                   Cross-dataset training/testing
            ├── train_on_both.yaml              Train with both MJN and MedleyDB datasets
            ├── train_on_medley.yaml            Train with MedleyDB dataset, test on both
            ├── train_on_mjn_inst.yaml          Train with MJN dataset, test on both (instrument cls model)
            └── train_on_mjn.yaml               Train with MJN dataset, test on both (channel cls model)
        └── inst_cls                        Instrument classification models (train and test with MJN dataset)
            ├── mjn_chavg.yaml                  The channel average model
            ├── mjn_chsep.yaml                  The channel attention model
            ├── mjn_from_mix.yaml               The from mixture only model
            └── mjn_guitar_seg.yaml             The segment-level guitar solo detection model
    ├── mert                            Mert code copied from its original repo
    ├── svm_baseline                    The SVM baseline for guitar solo detection
        ├── data_preparation.py             Prepare dataset for SVM model
        └── svm_guitar_segment.py           Train and evaluate the SVM model
    ├── test                            Some testing scripts for debugging
    ├── visualize                       For result visualization
        ├── confusion_matrix.py             Draw confusion matrix
        └── output_visualize.py             Draw multitrack heatmap, label, and prediction on a same figure
    ├── infer_segment.py                Infer on test set for segment-level guitar solo detection
    ├── infer.py                        Infer on test set for all other models
    ├── inst_utils_cross_dataset.py     Instrument utils when using two datasets together
    ├── inst_utils.py                   Instrument utils when only using MJN dataset
    ├── lightning_model.py              Define PyTorch Lightning models
    ├── lightning_test.py               Testing the model with Lightning model's test_step 
    ├── lightning_train_v100.py         Train the model with PyTorch Lightning (for V100 GPU)
    ├── lightning_train.py              Train the model with PyTorch Lightning (for RTX 3090 GPU)
    ├── Readme.md                       
    ├── requirements.txt                Required packages
    ├── solo_datasets.py                Datasets for model training/testing
    ├── solo_models.py                  Pytorch models for various detection tasks
    └── utils.py                        Utility functions

### Data directory
There are two data directories.

#### Local data dir
This directory is used for dataset annotation and constructing segment-level dataset from stem data.
It's located at

    MacBook: /Users/longshen/Code/Datasets
Please see `/Users/longshen/Code/Datasets/Readme.md` to know the structure of data directory.

#### Remote data dir
There is another directory that is used to save segment-level dataset for training, training results, model checkpoints. It's located at 

    Shannon: /home/longshen/data
Please see `/home/longshen/data/Readme.md` to know the structure of data directory.


## Prepare environment

### MacBook
    conda create -n solo python=3.12
    conda activate solo

    # Install pytorch 2.3.0 (any version)
    [Find the code on pytorch site]

    # Install dependencies
    pip install -r requirements.txt

### Linux Server
    
    Shannon and Neumann

    # A lower version of python is required because of low CUDA driver version on Shannon and Neumann
    conda create -n solo python=3.7
    conda activate solo

    # Install pytorch (a low version of torch is required due to low CUDA driver on Shannon and Neumann)
    pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
    
    # Need a patch for the torchaudio backend
    conda install 'ffmpeg<5' 

    # Install dependencies
    pip install -r requirements.txt
    

## Data Preprocessing

There are two major procedures to create dataset for training and testing. First, we annotate lead instrument for existing dataset. Then, we segment it into segment-level dataset for training.

### Data Annotation
#### MJN
0. Ensure the event recording's path is in `[local data dir]/MJN/Preliminary/metadata/raw_audio_dirs.xlsx`
1. Check the correctness of the stem-to-instrument definition file at `[local data dir]/MJN/Preliminary/inst_map`
2. Generate stem data by run the `mjn_get_stem_audio.py` under `data_preprocessing` directory.
3. Create Adobe Audition project at `[local data dir]/MJN/Preliminary/audition_prj`
4. Annotate in Adobe Audition, export the Markers to csv
5. Put the Markers file to `[local data dir]/MJN/Preliminary/annotations`

**NOTE**:
Some created stem audio are empty with only bleeding sound inside, which indicates the instruments was not used in a certain song or band performance. Such stems were deleted during annotation in Audition.

#### MedleyDB v1
1. Create the annotation Audition projects inside each song directory
2. Do the annotation
3. Export the Markers.csv to the song's directory

### Constructing segment-level dataset

Overall data preprocessing steps:
1. Create the metadata for segment-level dataset
2. Segment the audios to segment-level according to metadata
3. Split the dataset
4. Some final adjustment to the format of the metadata

Run `segment_dataset_mjn.py` and `segment_dataset_medley.py` to do these jobs.

Finally, after obtaining the segment-level dataset, we can do training and testing. It's better to move the data to the server because with GPU the training runs faster.

## Training and testing

Train the model with
    
    python lightning_train.py [path/to/corresponding/hparam]
    
    e.g.
    python lightning_train.py hparams/ch_cls/ch_permute.yaml    # train the final model

With the latest training script, the testing will be automatically executed after training finish. Testing will be perform twice, once on validation set, the second is on the testing set. Objective metrics will be computed for both datasets. The results will be shown in the command line with beautiful tables.

If for some reason you want to execute the testing again, do

    python lightning_test.py [path/to/corresponding/hparam]

## Visualize results

The lightning_test.py won't save models predictions so we need to write our own script to do this. First, run below commands to obtain model's output

    python infer.py [path/to/corresponding/hparam]
Note that there are two settings inside the `infer.py` you may need adjust before running the script. 
    
    chcls = True            # Whether the model being tested doing the channel classification
    split = 'valid'         # Do inference on validation or test set

Then, below commands can be used for visualization

    cd visualize
    
    # Confusion matrix
    python confusion_matrix.py [path/to/corresponding/hparam]

    # Multitrack audio heatmap with labels and predictions
    python output_visulize.py [path/to/corresponding/hparam]

**NOTE**: 
- The `split = 'valid'  # 'valid' or 'test'` inside confusion_matrix.py can control which split of dataset you would like to generate the confusion matrix for.
- The visualization results are saved to training log folders
- Due to performance reasons, the output_visulzie.py currently only draw for samples with change of lead instruments, so that it can finish running much quicker.

## Where can the performance be enhanced
- The currenct implementation combine the entire group of audio in the multitrack to a single tensor and feed to the audio encoder. When the number of channel is large (e.g., data in MedleyDB), we have to use too small batch size (e.g., 1) to make the training successfully running, which may harm the perfromance where the trained model can reach. A better way may be, inside the forward function of the model, to loop over the group of audio one by one to obtain the audio features, in this way we can make the memory consumption much less in training, make larger batch size possible, hence may obtain better training results.

    