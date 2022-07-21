### Prerequisites

- This project is implemented in Pytorch (>1.8). Thus please install Pytorch first.

- ctcdecode==0.4 [[parlance/ctcdecode]](https://github.com/parlance/ctcdecode)，for beam search decode.

- [Optional] sclite [[kaldi-asr/kaldi]](https://github.com/kaldi-asr/kaldi), install kaldi tool to get sclite for evaluation. After installation, create a soft link toward the sclite:    
  `ln -s PATH_TO_KALDI/tools/sctk-2.4.10/bin/sclite ./software/sclite`
  We also provide a python version evaluation tool for convenience, but sclite can provide more detailed statistics.

- [Optional] [SeanNaren/warp-ctc](https://github.com/SeanNaren/warp-ctc) At the beginning of this research, we adopt warp-ctc for supervision, and we recently find that pytorch version CTC can reach similar results.

### Data Preparation

1. Download the RWTH-PHOENIX-Weather 2014 Dataset [[download link]](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/). Our experiments based on phoenix-2014.v3.tar.gz.

2. After finishing dataset download, extract it to ./dataset/phoenix, it is suggested to make a soft link toward downloaded dataset.   
    ```bash
   ln -s PATH_TO_DATASET/phoenix2014-release ./dataset/phoenix2014
   ```
   For CSL dataset please link it this way   
    ```bash
   ln -s PATH_TO_DATASET/CSL ./dataset/csl
   ```
3. The original image sequence is 210x260, we resize it to 256x256 for augmentation. Run the following command to generate gloss dict and resize image sequence.     

   ```bash
   cd ./preprocess
   python dataset_preprocess.py --process-image --multiprocessing
   ```
4.  For CSL preprocessing please run this command below

   ```bash
   cd ./preprocess
   python dataset_preprocess_csl.py --process-image --multiprocessing
   ```

### Inference

​	To evaluate the pretrained model, run the command below：   
`python main.py --load-weights saved_model.pt --phase test`

### Training

The priorities of configuration files are: command line > config file > default values of argparse. To train the SLR model on phoenix14, run the command below:

`python main.py --work-dir PATH_TO_SAVE_RESULTS --config PATH_TO_CONFIG_FILE --device AVAILABLE_GPUS`

### Feature Extraction

We also provide feature extraction function to extract frame-wise features for other research purpose, which can be achieved by:

`python main.py --load-weights PATH_TO_PRETRAINED_MODEL --phase features `
