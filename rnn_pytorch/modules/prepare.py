import os
import subprocess
import platform
import numpy as np
from glob import glob
import pandas as pd
from tqdm import tqdm
import soundfile as snd
import scipy.io.wavfile as wave
import utils.util as util
import random
from utils.util_feat import add_noise


class DataPrepare:
    """
        This is a class for collecting information of Google Speech Command Dataset Version 2.
    """

    def __init__(self, args, config, dataset_path, testset_path, output_path):
        # Dataset Info
        self.val_list = None
        self.test_list = None
        self.label_list = None
        self.label_list_test = None
        self.sample_rate = config['dataset']['sample_rate']

        # Data Path
        self.dataset_path = dataset_path
        self.testset_path = testset_path
        self.noise_path = os.path.join(self.dataset_path, '_background_noise_')
        self.output_path = output_path

        print("Dataset Path: ", self.dataset_path)
        print("Testset Path: ", self.testset_path)
        print("Output Path: ", self.output_path)

        # Get Validation List
        with open(os.path.join(self.output_path, 'val_list.txt')) as f:
            val_list = f.readlines()
            self.val_list = [x.strip() for x in val_list]

        # Get Test List
        with open(os.path.join(self.output_path, 'test_list.txt')) as f:
            test_list = f.readlines()
            self.test_list = [x.strip() for x in test_list]

        # Get Label List
        self.label_list = pd.read_csv('./data/label_list_10.csv', index_col='keyword')
        self.label_list_test = pd.read_csv('./data/label_list_10_test.csv', index_col='keyword')

    def decompose_data_path(self, file_path):
        if platform.system() == 'Windows':
            separator = '\\'
        else:
            separator = '/'
        elem_path = file_path.split(separator)
        file_dir = separator.join(elem_path[:-1])
        keyword = elem_path[-2]
        local_path = keyword + '/' + elem_path[-1]
        file_name = elem_path[-1].split('.')[0]
        speaker_id = file_name.split('_')[0]
        ori_file_name = file_name.split('#')[0]
        try:
            snr = int(file_name.split('#')[2])
        except:
            snr = None
        return keyword, local_path, file_name, file_dir, speaker_id, ori_file_name, snr

    def create_silence(self):
        # Evaluate the balanced class number (Take 'yes' as example)
        yes_path = os.path.join(self.dataset_path, 'yes')
        all_yes_files = [y for x in os.walk(yes_path) for y in glob(os.path.join(x[0], '*.wav'))]
        n_yes = len(all_yes_files)
        print("::: There are %d yes samples in the dataset folder." % n_yes)

        # Create Silence Tracks
        silence_path = os.path.join(self.dataset_path, '_silence_')
        try:
            os.makedirs(silence_path)
        except:
            pass

        # Get noise file paths
        all_noise_files = [y for x in os.walk(self.noise_path) for y in glob(os.path.join(x[0], '*.wav'))]
        all_noise_files = sorted(all_noise_files)

        n_silence = n_yes
        print("::: Creating %d silence samples in the dataset folder." % n_silence)
        for i in tqdm(range(n_silence), desc='Create Silence'):
            np.random.seed(i)

            # Select Random Noise Track
            noise_file = np.random.choice(all_noise_files)
            _, _, file_name, _, _, _, _ = self.decompose_data_path(noise_file)

            # Read noise track
            noise, sample_rate = snd.read(noise_file)

            # Read source
            cut = int(sample_rate)

            # Sample random noise segment
            cut_low = np.random.randint(len(noise) - cut)

            # Create silence track
            signal_silence = np.array(noise[cut_low:cut_low + cut])

            # write result
            aug_file_path = os.path.join(silence_path, file_name + '_V{:02}.wav'.format(i))

            signal_silence *= 2 ** 15
            signal_silence = signal_silence.astype(np.int16)
            wave.write(aug_file_path, sample_rate, signal_silence)  # much faster

    def collect(self):
        random.seed(0)

        # Description File Column Titles
        columns = ['group', 'keyword', 'label', 'path']

        # Description File
        description_list = []
        description_file = 'description.csv'

        # Loop over folders in the dataset
        all_file_paths = [y for x in os.walk(self.dataset_path) for y in glob(os.path.join(x[0], '*.wav'))]
        if len(all_file_paths) == 0:
            raise RuntimeError("No dataset files are founded. Please check the path to the GSCDV2 dataset.")
        all_file_paths = random.sample(all_file_paths, len(all_file_paths))

        ##########################
        # Collect Train & Val Set
        ##########################
        # Remove unused paths
        selected_paths = []
        for file_path in tqdm(all_file_paths, desc='Remove useless paths'):
            keyword, local_path, file_name, file_dir, _, oridog_file_name, _ = self.decompose_data_path(file_path)

            # Skip background noise and test set
            if keyword == '_background_noise_' or local_path in self.test_list:
                continue
            else:
                selected_paths.append(file_path)
        selected_paths = random.sample(selected_paths, len(selected_paths))

        # Evaluate the balanced class number (Take 'yes' as example)
        yes_path = os.path.join(self.dataset_path, 'yes')
        all_yes_files = [y for x in os.walk(yes_path) for y in glob(os.path.join(x[0], '*.wav'))]
        n_yes = len(all_yes_files)
        print("::: There are %d yes samples in the dataset folder." % n_yes)

        # Get number of target samples
        max_n_unknown_train = n_yes
        max_n_unknown_val = 400
        n_silence_val = 400
        n_unknown_train = 0
        n_unknown_val = 0
        n_silence = 0
        n_silence_train = 0

        # Select Unknown
        list_unknown_keywords = ['backward', 'bed', 'bird', 'cat', 'dog', 'eight', 'five',
                                 'follow', 'forward', 'four', 'happy', 'house', 'learn',
                                 'marvin', 'nine', 'one', 'seven', 'sheila', 'six',
                                 'three', 'tree', 'two', 'visual', 'wow', 'zero']
        num_unknown_keywords = len(list_unknown_keywords)
        num_sample_per_unknown_keywords_train = np.ceil(float(max_n_unknown_train) / float(num_unknown_keywords)).astype(np.int64)
        num_sample_per_unknown_keywords_val = np.ceil(float(max_n_unknown_val) / float(num_unknown_keywords)).astype(np.int64)

        # Get Unknown Samples
        list_unkown_path_train = []
        list_unkown_path_val = []
        for keyword in list_unknown_keywords:
            unknown_path = os.path.join(self.dataset_path, keyword)
            all_unknown_paths = [y for x in os.walk(unknown_path) for y in glob(os.path.join(x[0], '*.wav'))]
            selected_unknown_paths = random.sample(all_unknown_paths, num_sample_per_unknown_keywords_train + num_sample_per_unknown_keywords_val)
            list_unkown_path_train.extend(selected_unknown_paths[0:num_sample_per_unknown_keywords_train])
            list_unkown_path_val.extend(selected_unknown_paths[num_sample_per_unknown_keywords_train:])

        # Collect Samples except unknown
        for idx, file_path in enumerate(tqdm(selected_paths, desc='Collect Train & Val')):
            keyword, local_path, file_name, file_dir, speaker_id, ori_file_name, snr = self.decompose_data_path(
                file_path)
            ori_local_path = keyword + '/' + ori_file_name + '.wav'
            # Assign Sets
            label = int(self.label_list.loc[keyword].to_numpy())
            if label == 1:
                continue

            if ori_local_path in self.val_list:
                group = 'val'
                # Limit the number of unknown keywords in the val set
                if label == 1:
                    if n_unknown_val > max_n_unknown_val:
                        continue
                    else:
                        n_unknown_val += 1

            else:
                group = 'train'
                # Add part of silence label into validation set
                if label == 0:
                    if n_silence < n_silence_val:
                        group = 'val'
                        n_silence += 1
                    else:
                        group = 'train'
                        n_silence_train += 1
                # Limit the number of unknown keywords
                if label == 1:
                    if n_unknown_train > max_n_unknown_train:
                        continue
                    else:
                        n_unknown_train += 1

            # Add a row to the description file
            row = {'group': group,
                   'keyword': keyword,
                   'label': label,
                   'path': file_path}
            description_list.append(row)
        print("Number of _silence_ in Train Set: ", n_silence_train)

        # Collect Samples except unknown
        for idx, file_path in enumerate(tqdm(list_unkown_path_train, desc='Collect Train Unknown')):
            group = 'train'
            keyword = '_unknown_'
            label = '1'
            # Add a row to the description file
            row = {'group': group,
                   'keyword': keyword,
                   'label': label,
                   'path': file_path}
            description_list.append(row)

        # Collect Samples except unknown
        for idx, file_path in enumerate(tqdm(list_unkown_path_val, desc='Collect Val Unknown')):
            group = 'val'
            keyword = '_unknown_'
            label = '1'
            # Add a row to the description file
            row = {'group': group,
                   'keyword': keyword,
                   'label': label,
                   'path': file_path}
            description_list.append(row)

        ##########################
        # Collect Test Set
        ##########################
        group = 'test'

        # Loop over folders in the dataset
        all_file_paths = [y for x in os.walk(self.testset_path) for y in glob(os.path.join(x[0], '*.wav'))]
        all_file_paths = sorted(all_file_paths)

        # Remove unused paths
        selected_paths = []
        for file_path in all_file_paths:
            keyword, local_path, file_name, file_dir, _, ori_file_name, snr = self.decompose_data_path(file_path)
            selected_paths.append(file_path)

        for idx, file_path in enumerate(tqdm(selected_paths, desc='Collect Test')):
            keyword, local_path, file_name, file_dir, speaker_id, ori_file_name, snr = self.decompose_data_path(
                file_path)
            label = int(self.label_list.loc[keyword].to_numpy())

            # Add a row to the description file
            row = {'group': group,
                   'keyword': keyword,
                   'label': label,
                   'path': file_path}
            description_list.append(row)

        ##########################
        # Write Description
        ##########################
        df = pd.DataFrame(description_list, columns=columns)
        df = df.sort_values(by=['group', 'label'], ascending=[False, False])
        description_file = os.path.join(self.output_path, description_file)
        df.to_csv(description_file, index=False)
