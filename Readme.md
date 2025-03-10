This repository is being updated.

# Lead Instrument Detection from Multitrack Music
[ [Paper](https://arxiv.org/pdf/2503.03232) | [GitHub Repo](https://github.com/Sonata165/LeadInstrumentDetection) ]

This is the official code repository of the ICASSP 2025 paper Lead Instrument Detection from Multitrack Music. The components of this repository include:
- Lead instrument annotation for the MedleyDB v1 dataset.
- Model's checkpoints
- Code to conduct training, evaluation, inference


## Directories

### Code Directory

The location where you clone this repository is the code directory. The structure is as follows:
    
    .
    ├── data_preprocessing              Scripts for creating metadata and segment-level datasets
    ├── hparams                         Hyperparameters/configurations for all solo detection models
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
    ├── lightning_train.py              Train the model with PyTorch Lightning (for RTX 3090 GPU)
    ├── Readme.md                       
    ├── requirements.txt                Required packages
    ├── solo_datasets.py                Datasets for model training/testing
    ├── solo_models.py                  Pytorch models for various detection tasks
    └── utils.py                        Utility functions

### Dataset Directory

The `datasets` folder serves as the dataset directory. After cloning this repository, move `datasets` to a hard drive with >5GB remaining spaces. After finishing the dataset preprocessing, the structure of the dataset directory should look like

    datasets
    └── MedleyDB     
        └── v1_segmented
            ├── data                    The folder containing all segment-level audios
                ├── AClassicEdu...      Song name (of the first song)
                    ├── 0               Segment ID
                    ├── 1               
                    ...
                ...
            ├── metadata_seg.json       Segment-level annotation. Created from raw annotation files
            ├── metadata_splitted.json  Added splitting info to segment-level annotation. Created from the file above.
            └── metadata.json           Annotation file used in training/evaluation. Reorganized from the file above.

## Prepare Environment

### MacBook
    conda create -n solo python=3.12
    conda activate solo

    # Install pytorch 2.3.0 (any version should be OK)
    [Find the code on pytorch site]

    # Install dependencies
    pip install -r requirements.txt

### Linux Server
    
    # The experiments were conducted on some old machines that require a low CUDA versions
    # But you may also try newer python, pytorch, and CUDA if your machine support them
    conda create -n solo python=3.7
    conda activate solo

    # Install pytorch (a low version of torch is required due to low CUDA driver on Shannon and Neumann)
    pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
    
    # May need the below patch for the torchaudio backend
    conda install 'ffmpeg<5' 

    # Install dependencies
    pip install -r requirements.txt


## Data Preprocessing

There are two major procedures to create dataset for training and testing. First, we annotate lead instrument for existing dataset. Then, we segment it into segment-level dataset for training.

### Data Annotation
Here are the procedures we adopted to annotate the MedleyDB v1 dataset.

1. Create the annotation Audition projects inside each song directory
2. Do the annotation
3. Export the Markers.csv to the song's directory


### Constructing segment-level dataset

Overall data preprocessing steps:
1. Create the metadata for segment-level dataset
2. Segment the audios to segment-level according to metadata
3. Split the dataset
4. Some final adjustment to the format of the metadata

Run `segment_dataset_medley.py` to do these jobs.

Finally, after obtaining the segment-level dataset, we can do training and testing. It's better to move the data to the server because with GPU the training runs faster.

## Training and Testing


Train the model with
    
    python lightning_train.py [path/to/corresponding/hparam]
    
    e.g.
    python lightning_train.py hparams/ch_cls/ch_permute.yaml    # train our final model

Note: below hparam file used the MJN dataset for training. You may need to adjust the dataset path to correctly train the model with MedleyDB.

With the latest training script, the testing will be automatically executed after training finish. Testing will be perform twice, once on validation set, the second is on the testing set. Objective metrics will be computed for both datasets. The results will be shown in the command line with beautiful tables.

If for some reason you want to execute the testing again, do

    python lightning_test.py [path/to/corresponding/hparam]

## Inference and Visualize Results

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


## Optimize VRAM
- The currenct implementation combine the entire group of audio in the multitrack to a single tensor and feed to the audio encoder. When the number of channel is large (e.g., data in MedleyDB), we have to use too small batch size (e.g., 1) to make the training successfully running, which may harm the perfromance where the trained model can reach. A better way may be, inside the forward function of the model, to loop over the group of audio one by one to obtain the audio features, in this way we can make the memory consumption much less in training, make larger batch size possible, hence may obtain better training results.


## Citation
Please cite our paper with below format:

    @inproceedings{ou2025lead,
        author    = {Longshen Ou and Yu Takahashi and Ye Wang},
        title     = {Lead Instrument Detection from Multitrack Music},
        booktitle = {Proceedings of the 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
        year      = {2025},
        organization = {IEEE}
    }
