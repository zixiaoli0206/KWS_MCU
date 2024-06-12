# PyTorch Framework RNN training and feature generatioon
PyTorch framework for training and feature generation. Note that this is not the core of this project and adapted from author's previous project

# Supported Dataset
* Google Speech Command Dataset Version 2 (--dataset_name gscdv2)
* Put the GSC dataset in the ./data/. Contact zixili@ethz.ch if you want to run the whole script with the required dataset

# Prerequisite
Install Miniconda
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
```

Create an environment using the following command:
```
conda create -n cochclass python=3.10 numpy matplotlib pandas tqdm h5py \
    scipy jupyter seaborn scikit-learn editdistance  \
    pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch-nightly -c nvidia
```

Activate the environment before running the script.
```
conda activate cochclass
```

# Run
Navigate to the project folder and run experiments with the main.py file by specifying the target dataset. 
Run the code with --step to run the specific step. For example:
# Prepare Dataset
```
python main.py --dataset_name gscdv2 --step prepare
```
# Feature Extraction
The feature configuration files are under ./config
```
python main.py --dataset_name gscdv2 --step feature
```
# Pretrain
Pretrain using GRU.
Hyperparameters are defined in ./modules/arguments.py
```
python main.py --dataset_name gscdv2 --step pretrain
```

# Acknowledgements
This project is adapted from [Chang Gao's DeltaRNN project](https://github.com/gaochangw/DeltaRNN). 

Please refer to the original repositories for more detailed information on licensing and usage terms.