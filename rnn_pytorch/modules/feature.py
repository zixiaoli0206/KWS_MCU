import math
import os
import numpy as np
import pandas as pd
from utils import util_feat, util
from modules import log
import utils.feature.digital_afe as digital_fex
from tqdm import tqdm
from utils.util_feat import write_dataset


def align(x):
    aligned = []
    len = x.shape[1]
    for i in x:
        all_mean = []
        for t in range(0, len-61):
            sec = i[t:t+61, :]
            all_mean.append(np.mean(sec))
        all_mean = np.asarray(all_mean)
        start_index = np.argmax(all_mean)
        aligned.append(i[start_index:start_index+61])
    aligned = np.stack(aligned, axis=0)
    return aligned


class FeatExtractor:
    def __init__(self, config, args, path_root):
        self.args = args
        self.path_root = path_root
        self.dataset_name = args.dataset_name
        self.description_name = 'description.csv'
        self.feat_type = config["feature"]['type']
        self.analog_feature_config = config["feature"]['analog']
        self.digital_feature_config = config["feature"]['digital']
        self.config = config
        # Load dataframe
        self.df_description = pd.read_csv(os.path.join(self.path_root, 'data', self.description_name))

    def extract(self):

        # Initialization
        np.random.seed(2)

        X_train = []
        X_val = []
        X_test = []

        f_train = []
        f_val = []
        f_test = []

        y_train = []
        y_val = []
        y_test = []

        # Get File ID
        _, trainfile, valfile = log.gen_trainset_name(self.args, self.config)
        testfile = log.gen_testset_name(self.args, self.config)

        # Dataset Path
        dataset_path, testset_path = util.get_dataset_path(self.args)

        # Get Filters
        if self.feat_type == 'analog':
            fbank, filt_b, filt_a, hz_points = util_feat.get_aafe_filters(self.args, self.config, self.df_description,
                                                                          self.args.plt_feat)
        else:
            fbank, filt_b, filt_a, hz_points = util_feat.get_dafe_filters(self.args, self.config, self.df_description,
                                                                          self.args.plt_feat)

        # Loop over dataframe
        for row in tqdm(self.df_description.itertuples(), total=self.df_description.shape[0]):
            filepath = row.path
            filepath = filepath.replace("\\", "/")
            filepath = filepath.replace("C:/", "/")

            # Digital Audio Front End (Log Filter Bank)
            if self.feat_type == 'digital':
                # Feature
                features, frames, sample_rate = digital_fex.extract_feat(path=filepath,
                                                                         config=self.config,
                                                                         fbank=fbank,
                                                                         MFCC=False,
                                                                         plot=self.args.plt_feat)
                dim_T = features.shape[0]
                len_label = int(math.floor(dim_T * self.config['feature']['delta_t']))
                label = np.zeros((dim_T, 1))
                label[-len_label:, :] = int(row.label)
                flag = int(row.label)

                # Use VAD to label the samples
                if self.args.use_vad:
                    import webrtcvad
                    vad = webrtcvad.Vad()
                    vad.set_mode(3)

                    label = np.zeros((dim_T,))
                    for i in range(dim_T):
                        label[i] = int(vad.is_speech(frames[i, :], int(sample_rate)))
                    pos_label_idx = np.argwhere(label == 1)
                    len_voice = pos_label_idx.shape[0]
                    len_label = int(math.floor(len_voice * 0.9))
                    if len_voice != 0:
                        dec_window_start_idx = int(pos_label_idx[-1] - self.args.delta_t)
                        # dec_window_start_idx = int(pos_label_idx[0])
                        dec_window_end_idx = int(pos_label_idx[-1])
                        # dec_window_end_idx = int(pos_label_idx[-1] + delta_t)
                        label = np.zeros((dim_T, 1))
                        # print(int(row.label))
                        label[dec_window_start_idx:dec_window_end_idx, :] = int(row.label) + 1
                        flag = int(row.label) + 1
                    else:
                        continue
                        # label = np.zeros((dim_T, 1))
                        # flag = 0

            if row.group == 'train':
                X_train.append(features)
                y_train.append(label)
                f_train.append(flag)
            elif row.group == 'val':
                X_val.append(features)
                y_val.append(label)
                f_val.append(flag)
            elif row.group == 'test':
                X_test.append(features)
                y_test.append(label)
                f_test.append(flag)

        # Process Datasets
        dataset_train = self.get_dataset(X_train, y_train, f_train)
        dataset_val = self.get_dataset(X_val, y_val, f_val)
        dataset_test = self.get_dataset(X_test, y_test, f_test)

        # Write Dataset
        write_dataset(os.path.join(self.path_root, 'feat', trainfile), dataset_train)
        write_dataset(os.path.join(self.path_root, 'feat', valfile), dataset_val)
        write_dataset(os.path.join(self.path_root, 'feat', testfile), dataset_test)

        print("Feature stored in: ", os.path.join(self.path_root, 'feat'))
        print("Feature Extraction Completed...                                     ")
        print(" ")

    def get_dataset(self, x, y, f):
        features = np.concatenate(x, axis=0).astype(np.float32)
        feature_lengths = np.asarray([sample.shape[0] for sample in x]).astype(np.int32)
        targets = np.concatenate(y, axis=0).astype(np.int32)
        target_lengths = np.asarray([len(target) for target in y]).astype(np.int32)
        flag = np.stack(f, axis=0).astype(np.int32)
        n_features = features.shape[-1]
        n_classes = np.max(flag).astype(np.int32) + 1
        dict_dataset = {'features': features, 'feature_lengths': feature_lengths, 'targets': targets,
                        'target_lengths': target_lengths, 'flag': flag, 'n_features': n_features,
                        'n_classes': n_classes}
        return dict_dataset

    def get_stream_dataset(self, x, y):
        features = x.astype(np.float64)
        targets = y.astype(np.int64)
        n_features = features.shape[-1]
        n_classes = np.max(y).astype(np.int32) + 1
        dict_dataset = {'features': features, 'targets': targets, 'n_features': n_features,
                        'n_classes': n_classes}
        return dict_dataset
