import math
import numpy as np
import h5py
import utils.util as util
from scipy import optimize
from utils.util import load_h5py_data, log_lut
import matplotlib.pyplot as plt
from joblib import dump

def calculate_noise(x, snr_db=20):
    snr = 10 ** (snr_db / 10.0)
    # Calculate signal power and convert to dB
    sig_avg_watts = np.mean(np.power(np.abs(x), 2))
    noise_avg_watts = sig_avg_watts / snr
    # noise_avg_watts_actual = np.mean(np.power(np.abs(noise_volts), 2))
    return noise_avg_watts


def add_noise(x, noise_avg_watts):
    # Generate an sample of white noise
    num_sample = x.shape[0]
    feat_size = x.shape[1]
    noise_volts = np.random.normal(loc=0, scale=np.sqrt(noise_avg_watts),
                                   size=(num_sample, feat_size))
    return x + noise_volts


def noise_augment(x, snr_db=20):
    snr = 10 ** (snr_db / 10.0)
    # Calculate signal power and convert to dB
    sig_avg_watts = np.mean(np.power(np.abs(x), 2))
    noise_avg_watts = sig_avg_watts / snr
    # Generate an sample of white noise
    num_sample = x.shape[0]
    feat_size = x.shape[1]
    noise_volts = np.random.normal(loc=0, scale=np.sqrt(noise_avg_watts),
                                   size=(num_sample, feat_size))
    noise_avg_watts_actual = np.mean(np.power(np.abs(noise_volts), 2))
    out = x + noise_volts
    return out


def plot_feature(x, path, name, label, keyword, vmin=0, vmax=8000):
    fig, axes = plt.subplots(1, 1, figsize=(12, 6))
    im1 = axes.imshow(x.T, aspect='auto')
    im1.set_clim(vmin=vmin, vmax=vmax)
    plt.colorbar(im1)
    title = "Name: %s | Label: %2d | Keyword: %s" % (name, label, keyword)
    axes.tick_params(labelsize=16)
    axes.set_xlabel('Frames', fontsize=18)
    axes.set_ylabel('Channels', fontsize=18)
    axes.set_title(title, fontsize=18)
    # axes.set_xticks(np.arange(0, 101, 1))
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    # plt.show()


def mask_channel(set):
    mask = np.ones(16).astype(np.float32)
    mask[13] = 0
    # mask[7] = 0
    # mask[8] = 0
    set = set * mask
    return set


###########################################################################################################
# Google Speech Command Dataset
###########################################################################################################
class CustomDataLoader(object):
    def __init__(self, args, config, **kwargs):
        self.trainfile = args.trainfile
        self.valfile = args.valfile
        self.testfile = args.testfile
        self.num_sample_train = 0
        self.num_sample_val = 0
        self.num_sample_test = 0
        self.mean_train_feat = 0
        self.std_train_feat = 0
        self.zero_padding = args.zero_padding
        self.feat_type = config['feature']['type']
        self.bw_feat = config['feature']['bw_feat']
        self.qa = config['model']['qa']
        self.aqi = config['model']['aqi']
        self.aqf = config['model']['aqf']
        self.snr = args.snr

        self.double_dynamic_range = config['dataloader']['double_dynamic_range']

        self.std_train_preprocessed_rec = None
        # Quantize Features
        try:
            self.qf = kwargs['qf']
        except:
            self.qf = config['feature']['qf']
        # Use LUT-Log Funtion
        try:
            self.log_feat = kwargs['log_feat']
        except:
            self.log_feat = config['feature']['log_feat']
        try:
            self.normalization = kwargs['normalization']
        except:
            self.normalization = config['feature']['normalization']
        try:
            self.remove_offset = kwargs['remove_offset']
        except:
            self.remove_offset = config['feature']['remove_offset']
        try:
            self.scale_channels = kwargs['scale_channels']
        except:
            self.scale_channels = config['feature']['scale_channels']
        # Test with different volume
        self.test_vol_db = args.test_vol_db

        # Piecewise Linear Log
        x_log = np.arange(0, 4096, 1).astype(np.float64)
        y_log = np.log(x_log + 1)
        popt, pcov = optimize.curve_fit(util.piecewise_linear_log, x_log, y_log)
        self.popt = util.quantize_array(popt, 4, 16, 1)
        self.approx_log = config['feature']['approx_log']

        # Evaluate train set
        self.train = None
        self.val = None
        self.test = None

        with h5py.File(self.trainfile, 'r') as hf:
            self.train_dict_data = load_h5py_data(hf)
        with h5py.File(self.valfile, 'r') as hf:
            self.dev_dict_data = load_h5py_data(hf)
        with h5py.File(self.testfile, 'r') as hf:
            self.test_dict_data = load_h5py_data(hf)

        self.n_features = int(self.test_dict_data['n_features'].astype(int))
        self.n_classes = int(self.test_dict_data['n_classes'].astype(int))

        # Get trainset info
        self.train = self.train_dict_data['features'].astype(np.float32)

        # Stat for each feature
        self.mean_train_feat = np.mean(self.train, axis=0)
        self.std_train_feat = np.std(self.train, axis=0)
        self.max_train_feat = np.amax(self.train, axis=0)
        self.min_train_feat = np.amin(self.train, axis=0)

        # Stat for all features
        self.shape_train = self.train.shape
        self.mean_train = np.mean(self.train)
        self.std_train = np.std(self.train)
        self.median_train = np.median(self.train)
        self.max_train = np.amax(self.train)
        self.min_train = np.amin(self.train)

        # Generate Quantized Features
        self.train = self.gen_qfeature(features=self.train, amax=self.max_train)

        # Log
        self.train = self.preprocess_feature(features=self.train,
                                             set_name='train',
                                             log_feat=self.log_feat)
        # Stat of preprocessed features
        self.mean_train_preprocessed = np.mean(self.train)
        self.std_train_preprocessed = np.std(self.train)
        self.median_train_preprocessed = np.median(self.train)
        self.max_train_preprocessed = np.amax(self.train)
        self.min_train_preprocessed = np.amin(self.train)
        self.mean_train_preprocessed_feat = np.mean(self.train, axis=0)
        self.std_train_preprocessed_feat = np.std(self.train, axis=0)

    def process_dataset(self, set_name):
        if set_name == 'train':
            self.train = self.train_dict_data['features'].astype(np.float32)
            self.train_feature_lengths = self.train_dict_data['feature_lengths'].astype(int)
            self.train_targets = np.squeeze(self.train_dict_data['targets'].astype(int))
            self.train_target_lengths = self.train_dict_data['target_lengths'].astype(int)
            self.train_flag = self.train_dict_data['flag'].astype(int)
            self.num_sample_train = self.train_feature_lengths.size

            # Add Noise
            if self.snr != 0:
                self.noise_level_train = calculate_noise(self.train, snr_db=self.snr)
                self.train = add_noise(self.train, self.noise_level_train)

            self.train = self.gen_qfeature(features=self.train, amax=self.max_train)

            # Log Function
            self.train = self.preprocess_feature(features=self.train,
                                                 set_name='train',
                                                 log_feat=self.log_feat)

            # Postprocess features
            self.train = self.normalize_feature(self.train)
            self.train_norm_max = np.amax(self.train)
            self.train_norm_min = np.amin(self.train)
            self.train_norm_mean = np.mean(self.train)
            self.train_norm_std = np.std(self.train)
            self.train = util.quantize_array(self.train, self.aqi, self.aqf, self.qa)
            dump(self.train_dict_data, './data/MCU_dataset/train_dict_data.joblib')
            dump(self.train, './data/MCU_dataset/train_feat.joblib')

        if set_name == 'val':
            # Evaluate val set
            self.val = self.dev_dict_data['features'].astype(np.float32)
            # self.val = mask_channel(self.val)
            self.val_feature_lengths = self.dev_dict_data['feature_lengths'].astype(int)
            self.val_targets = np.squeeze(self.dev_dict_data['targets'].astype(int))
            self.val_target_lengths = self.dev_dict_data['target_lengths'].astype(int)
            self.val_flag = self.dev_dict_data['flag'].astype(int)
            # Add Noise
            if self.snr != 0:
                self.val = add_noise(self.val, self.noise_level_train)
            # Stat for each feature
            self.mean_val_feat = np.mean(self.val[:, :], axis=0)
            self.std_val_feat = np.std(self.val[:, :], axis=0)
            self.max_val_feat = np.amax(self.val[:, :], axis=0)
            self.min_val_feat = np.amin(self.val[:, :], axis=0)
            # Stat for all features
            self.shape_val = self.val.shape
            self.mean_val = np.mean(self.val[:, :])
            self.median_val = np.median(self.val[:, :])
            self.std_val = np.std(self.val[:, :])
            self.max_val = np.amax(self.val[:, :])
            self.min_val = np.amin(self.val[:, :])
            self.num_sample_val = self.val_feature_lengths.size

            # Process features
            self.val = self.gen_qfeature(features=self.val, amax=self.max_train)
            self.val = self.preprocess_feature(features=self.val,
                                               set_name='val',
                                               log_feat=self.log_feat)
            self.val = self.normalize_feature(self.val)
            self.val = util.quantize_array(self.val, self.aqi, self.aqf, self.qa)
            dump(self.dev_dict_data, './data/MCU_dataset/val_dict_data.joblib')
            dump(self.val, './data/MCU_dataset/val_feat.joblib')

        if set_name == 'test':
            # Evaluate test set
            self.test = self.test_dict_data['features'].astype(np.float32)
            # self.test = mask_channel(self.test)
            self.test_feature_lengths = self.test_dict_data['feature_lengths'].astype(int)
            self.test_targets = np.squeeze(self.test_dict_data['targets'].astype(int))
            self.test_target_lengths = self.test_dict_data['target_lengths'].astype(int)
            self.test_flag = self.test_dict_data['flag'].astype(int)

            # Add Noise
            # if self.snr != 0:
            #     self.test = add_noise(self.test, self.noise_level_train)
            # Stat for each feature
            self.mean_test_feat = np.mean(self.test[:, :], axis=0)
            self.std_test_feat = np.std(self.test[:, :], axis=0)
            self.max_test_feat = np.amax(self.test[:, :], axis=0)
            self.min_test_feat = np.amin(self.test[:, :], axis=0)
            # Stat for all features
            self.shape_test = self.test.shape
            self.mean_test = np.mean(self.test[:, :])
            self.median_test = np.median(self.test[:, :])
            self.std_test = np.std(self.test[:, :])
            self.max_test = np.amax(self.test[:, :])
            self.min_test = np.amin(self.test[:, :])
            self.num_sample_test = self.test_feature_lengths.size

            self.test = self.gen_qfeature(features=self.test, amax=self.max_train)

            self.test = self.preprocess_feature(features=self.test,
                                                set_name='test',
                                                log_feat=self.log_feat)
            # Stat of preprocessed features
            self.mean_test_preprocessed = np.mean(self.test)
            self.std_test_preprocessed = np.std(self.test)
            self.test = self.normalize_feature(self.test)
            self.test = util.quantize_array(self.test, self.aqi, self.aqf, self.qa)
            dump(self.test_dict_data, './data/MCU_dataset/test_dict_data.joblib')
            dump(self.test, './data/MCU_dataset/test_feat.joblib')

    def gen_qfeature(self, features, amax):
        new_features = np.array(features)
        if self.qf:
            # Get Range of Quantized Features
            new_features_max = amax

            # Map Features to [0, 1]
            new_features /= new_features_max

            # Cast Features to Fixed Point Numbers
            if self.double_dynamic_range:
                new_features = np.around(new_features * (2 ** (self.bw_feat + 1) - 1))
            else:
                new_features = np.around(new_features * (2 ** self.bw_feat - 1))  # correct

        return new_features

    def scale(self, features):
        if self.scale_channels and self.channel_scale is not None:
            features = self.channel_scale_q * features  # Scale channels
            features = util.quantize_array(x=features, qi=self.bw_feat, qf=0, enable=1, unsigned=1)
        return features

    def preprocess_feature(self, set_name, features, log_feat):
        new_features = np.array(features)
        try:
            if self.test_vol_db != 0 and set_name == 'test':
                # Get amplification factor for signal.
                amp_factor = np.sqrt(10.0 ** (float(self.test_vol_db) / 10.0))
                new_features *= amp_factor
                print("Test Set Amplified by %f times..." % amp_factor)
        except AttributeError:
            pass

        # if self.channel_scale is not None and set_name == 'test':

        # if self.custom_data != '':
        #     channel_max = np.max(new_features)
        #     channel_dr_scale = 4095 / channel_max
        #     new_features *= channel_dr_scale
        if log_feat:
            if self.approx_log:
                new_features = log_lut(new_features, qi_in=self.bw_feat + 1, qf_in=0, qi_out=3, qf_out=8)
                # new_features = utils.piecewise_linear_log(new_features + 1, *self.popt)
            else:
                new_features = np.log2(new_features + 1)

        return new_features

    def normalize_feature(self, features):
        # eps = 1e-5
        if self.normalization == 'standard_feat':
            if self.qf:
                features -= util.quantize_array(self.mean_train_preprocessed_feat[np.newaxis, :], 8, 8, 1)
                features = util.quantize_array(features, 8, 8, 1)
                features *= util.quantize_array(1 / (self.std_train_preprocessed_feat[np.newaxis, :]), 8, 8, 1)
                features = util.quantize_array(features, self.aqi, self.aqf, self.qa)
            else:
                features -= self.mean_train_feat[np.newaxis, :]
                features /= self.std_train_feat[np.newaxis, :]
        elif self.normalization == 'standard':
            # if self.custom_data != '':
            #     mean = self.mean_test_preprocessed
            #     std = self.std_test_preprocessed
            # else:
            mean = self.mean_train_preprocessed
            std = self.std_train_preprocessed
            # mean = self.mean_test_preprocessed
            # std = self.std_test_preprocessed
            if self.qf:
                mean = util.quantize_array(mean, 8, 8, 1)
                features -= mean
                # print(features)
                features = util.quantize_array(features, 8, 8, 1)
                self.std_train_preprocessed_rec = util.quantize_array(1 / std, 8, 8, 1)
                features *= self.std_train_preprocessed_rec
                # features /= std
                # features *= 12
                # features = util.quantize_array(features, self.aqi, self.aqf, self.qa)
            elif self.log_feat:
                mean = self.mean_train_preprocessed
                std = self.std_train_preprocessed
                features -= mean
                features /= std
            else:
                features -= self.mean_train
                features /= self.std_train
        elif self.normalization == 'min_max':
            features -= self.min_train_feat[np.newaxis, :]
            features /= (self.max_train_feat[np.newaxis, :] - self.min_train_feat[np.newaxis, :])
        else:
            print('Warning: Features are not normalized.')
            return features

        return features

    def get_num_batch(self, set_name, batch_size):
        if set_name == 'train':
            return int(math.ceil(self.num_sample_train / batch_size))
        elif set_name == 'val':
            return int(math.ceil(self.num_sample_val / batch_size))
        elif set_name == 'test':
            return int(math.ceil(self.num_sample_test / batch_size))

    def get_num_sample(self, set_name):
        if set_name == 'train':
            return self.num_sample_train
        elif set_name == 'val':
            return self.num_sample_val
        elif set_name == 'test':
            return self.num_sample_test

    def add_grad(self, batch):
        batch_new = []
        for sample in batch:
            grad_first_order = np.gradient(sample, edge_order=1, axis=0)
            grad_second_order = np.gradient(grad_first_order, edge_order=1, axis=0)
            grad_first_order = util.quantize_array(grad_first_order, 1, self.bw_feat - 1, self.qf)
            grad_second_order = util.quantize_array(grad_second_order, 1, self.bw_feat - 1, self.qf)
            sample_new = np.hstack((sample, grad_first_order, grad_second_order))
            batch_new.append(sample_new)
        return batch_new

    def iterate(self,
                epoch,
                set_name,
                batch_size=32,
                shuffle_type='high_throughput',
                feat_grad=0,
                enable_gauss=0,
                mode='CTC'):
        """
        :param epoch:         The index of current epoch to set random seed
        :param set_name:           Dataset to iterate - 'train', 'val' or 'test'
        :param batch_size:    Batch size
        :param shuffle_type:  Shuffling of batch data
        :param feat_grad:     Append 1st & 2nd order gradient of feature to itself
        :param enable_gauss:  Add gaussian noise to the feature
        :param mode:          Mode of iteration
        :return:              Yield iterator
        """

        np.random.seed(epoch)

        self.process_dataset(set_name=set_name)

        if mode == 'CTC':
            ctc_label_shift = 0
        elif mode == 'Aligned':
            ctc_label_shift = 0
        else:
            ctc_label_shift = 0

        # Select datasets
        if set_name == 'train':
            features = self.train
            feature_lengths = self.train_feature_lengths
            targets = self.train_targets
            target_lengths = self.train_target_lengths
            flag = self.train_flag
        elif set_name == 'val':
            features = self.val
            feature_lengths = self.val_feature_lengths
            targets = self.val_targets
            target_lengths = self.val_target_lengths
            flag = self.val_flag
        elif set_name == 'test':
            features = self.test
            feature_lengths = self.test_feature_lengths
            targets = self.test_targets
            target_lengths = self.test_target_lengths
            flag = self.test_flag
        else:
            raise RuntimeError('Please select a valid set.')

        # Get index slices for features and targets
        feature_slice_idx = self.idx_to_slice(feature_lengths)
        label_slice_idx = self.idx_to_slice(target_lengths)

        # Dimensions
        # N - Number of samples
        # T - Sequence length
        # F - Feature size
        dim_N = feature_lengths.shape[0]
        dim_F = features.shape[1]

        # Batch Shuffle
        if shuffle_type == 'high_throughput':
            s_idx = np.argsort(feature_lengths)[::-1]
        elif shuffle_type == 'random':
            s_idx = np.random.permutation(dim_N)
        else:
            s_idx = range(dim_N)

        # Create a list of batch sample indices
        batches_idx = self.create_batch_idx(s_idx=s_idx, batch_size=batch_size)
        n_batches = len(batches_idx)  # Number of batches

        # Generate a batch iterator
        b = 0
        while b < n_batches:
            curr_batch_idx = batches_idx[b]

            # Load batch
            batch_feats = []
            batch_labels = []
            for sample_idx in curr_batch_idx:
                batch_feats.append(features[self.slc(sample_idx, feature_slice_idx), :])
                batch_labels.append(targets[self.slc(sample_idx, label_slice_idx)])

            # Add Feature Gradient
            if feat_grad:
                batch_feats = self.add_grad(batch_feats)

            # Add gaussian noise:
            if enable_gauss != 0.0:
                batch_feats = self.add_gaussian_noise(batch_feats, sigma=enable_gauss)

            # Zero Padding
            max_len = np.max(feature_lengths[curr_batch_idx])

            bX = np.zeros((len(curr_batch_idx), max_len, dim_F), dtype='float32')
            if mode == 'CTC':
                bY = []
            elif mode == 'Aligned':
                bY = np.zeros((len(curr_batch_idx), max_len), dtype='float32')
            else:
                bY = np.zeros((len(curr_batch_idx), max_len), dtype='float32')
            b_lenX = feature_lengths[curr_batch_idx].astype('int32')
            b_lenY = target_lengths[curr_batch_idx].astype('int32')
            b_flag = flag[curr_batch_idx].astype('int32')

            for i, sample in enumerate(batch_feats):
                if self.zero_padding == 'head':
                    bX[i, -b_lenX[i]:, :] = sample
                else:
                    bX[i, :b_lenX[i], :] = sample
                if mode == 'CTC':
                    ctc_labels = np.asarray(batch_labels[i]) + ctc_label_shift  # Make label 0 the 'blank'
                    bY.extend(ctc_labels)
                elif mode == 'Aligned':
                    bY[i, :b_lenX[i]] = batch_labels[i].squeeze()
                else:
                    bY[i, :b_lenX[i]] = batch_labels[i].squeeze()
            b += 1

            # features = bX
            # feature_0 = features[64]
            # plot_feature(feature_0)

            dict_batch_data = {'features': bX, 'feature_lengths': b_lenX, 'targets': bY, 'target_lengths': b_lenY,
                               'flag': b_flag}

            yield dict_batch_data

    def idx_to_slice(self, lengths):
        """
        Get the index range of samples
        :param lengths: 1-D tensor containing lengths in time of each sample
        :return: A list of tuples containing the start & end indices of each sample
        """
        idx = []
        lengths_cum = np.cumsum(lengths)
        for i, len in enumerate(lengths):
            start_idx = lengths_cum[i] - lengths[i]
            end_idx = lengths_cum[i]
            idx.append((start_idx, end_idx))
        return idx

    def slc(self, i, idx):
        return slice(idx[i][0], idx[i][1])

    def add_gaussian_noise(self, batch_feats, sigma=0.6):
        batch_gauss = []
        for sample in batch_feats:
            noise_mat = sigma * np.random.standard_normal(sample.shape)
            sample = sample + noise_mat

            batch_gauss.append(sample)
        return batch_gauss

    def create_batch_idx(self, s_idx, batch_size):
        list_batches = []
        batch = []

        for i, sample in enumerate(s_idx):
            if len(batch) < batch_size:
                batch.append(sample)
            else:
                list_batches.append(batch)
                batch = [sample]

        list_batches.append(batch)
        return list_batches
