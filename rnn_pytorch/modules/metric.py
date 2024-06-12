import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize

def gen_meter_args(args, n_classes, **kwargs):
    dict_meter_args = {'n_classes': n_classes, 'smooth': args.smooth,
                       'smooth_window_size': args.smooth_window_size,
                       'confidence_window_size': args.confidence_window_size, 'zero_padding': args.zero_padding,
                       'fire_threshold': args.fire_threshold, 'blank': 0, 'idx_silence': 0, 'threshold': 0}
    if args.dataset_name == 'timit':
        dict_meter_args['phn'] = args.phn
    return dict_meter_args


def slide_window(seq, window_size, window_stride):
    """
    :param seq: 2-D (T, C) tensor that is the real sequence
    :return seq_window: 3-D (T, window_size, C) tensor that is the slide window of a real sequence
    """
    seq_len = seq.shape[0]
    seq_window = torch.cat([seq[t:t + window_size, :] for t in np.arange(0, seq_len - window_size, window_stride)],
                           dim=0)
    return seq_window


# def get_smoothed_seq(seq, window_size, activation='softmax'):
#     """
#     :param seq: 2-D (T, C) tensor that is the real sequence
#     :param window_size: Size of the sliding window
#     :param activation: Activation function that converts logits to scores
#     :return: smoothed_seq: 2-D (T, C) tensor that is the smoothed sequence
#     """
#     if activation == 'log_softmax':
#         post_seq = F.log_softmax(seq, dim=-1)
#     elif activation == 'softmax':
#         post_seq = F.softmax(seq, dim=-1)
#     elif activation == 'sigmoid':
#         post_seq = torch.sigmoid(seq)
#     else:
#         post_seq = seq
#     # post_seq = torch.sigmoid(seq)
#     seq_len = post_seq.shape[0]
#     num_class = post_seq.shape[1]
#     padded_seq = torch.cat((torch.zeros(window_size - 1, num_class), post_seq), dim=0)
#     # windowed_seq: 3-D (T, window_size, C)
#     windowed_seq = torch.stack([padded_seq[t:t + window_size, :] for t in np.arange(0, seq_len)], dim=0)
#     sum_of_window = torch.sum(windowed_seq, dim=1)
#     denominator = torch.ones(seq_len, 1) * window_size
#     denominator[:min(window_size, seq_len), :] = torch.arange(1, min(window_size + 1, seq_len + 1)).view(-1, 1)
#     smoothed_seq = sum_of_window / denominator
#
#     return smoothed_seq
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def get_smoothed_seq(seq, window_size, activation='softmax'):
    """
    :param seq: 2-D (T, C) tensor that is the real sequence
    :param window_size: Size of the sliding window
    :param activation: Activation function that converts logits to scores
    :return: smoothed_seq: 2-D (T, C) tensor that is the smoothed sequence
    """
    from scipy.special import log_softmax, softmax
    if activation == 'log_softmax':
        post_seq = log_softmax(seq, axis=-1)
    elif activation == 'softmax':
        post_seq = softmax(seq, axis=-1)
    elif activation == 'sigmoid':
        post_seq = sigmoid(seq)
    else:
        post_seq = seq
    # post_seq = torch.sigmoid(seq)
    seq_len = post_seq.shape[0]
    num_class = post_seq.shape[1]
    zero_pad = np.zeros((window_size, num_class))
    padded_seq = np.concatenate((zero_pad, post_seq), axis=0)
    # windowed_seq: 3-D (T, window_size, C)
    windowed_seq = np.stack(
        [padded_seq[t - window_size:t + 1, :] for t in np.arange(window_size, seq_len + window_size)], axis=0)
    smoothed_seq = np.mean(windowed_seq, axis=1)
    return smoothed_seq


def get_score(seq, window_size, log_softmax=False):
    """
    :param seq: 2-D (T, C) tensor that is the real sequence
    :param window_size: Size of the sliding window
    :return: smoothed_seq: 2-D (T, C) tensor that is the smoothed sequence
    """
    if log_softmax:
        post_seq = torch.abs(F.log_softmax(seq, dim=-1))
    else:
        post_seq = F.softmax(seq, dim=-1)

    seq_len = post_seq.shape[0]
    num_class = post_seq.shape[1]
    padded_seq = torch.cat((torch.zeros(window_size - 1, num_class), post_seq), dim=0)
    # windowed_seq: 3-D (T, window_size, C)
    windowed_seq = torch.stack([padded_seq[t:t + window_size, :] for t in np.arange(0, seq_len)], dim=0)
    # prod_of_window: 2-D (T, C)
    prod_of_window = torch.prod(windowed_seq, dim=1)
    score = np.power(prod_of_window, 1.0 / float(window_size))
    score = 1 / score

    return score


def get_confidence_seq(seq, window_size):
    """
    :param seq: 2-D (T, C) tensor that is the real sequence
    :param window_size: Size of the sliding window
    :return: smoothed_seq: 2-D (T, C) tensor that is the smoothed sequence
    """
    seq_len = seq.shape[0]
    num_kw = seq.shape[1] - 1
    seq_kw = seq[:, 1:]
    padded_seq_kw = torch.cat((torch.zeros(window_size - 1, num_kw), seq_kw), dim=0)
    # windowed_seq: 3-D (T, window_size, C)
    window_seq = torch.stack([padded_seq_kw[t:t + window_size, :] for t in np.arange(0, seq_len)], dim=0)
    window_seq = window_seq.numpy().astype(np.float64)
    # max_prob_window_seq: 2-D (T, C)
    max_prob_window_seq = np.amax(window_seq, axis=1)
    # max_prob_window_seq = max_prob_window_seq.numpy()
    # prod_of_window: 1-D (T)
    prod_of_window = np.prod(max_prob_window_seq, axis=-1)
    confidence_seq = np.power(prod_of_window, 1.0 / float(num_kw))
    confidence_seq = torch.from_numpy(confidence_seq)
    return confidence_seq


def greedy_decoder(post_seq):
    """
    Greedy decoder with threshold
    :param post_seq: 2-D (T, C) tensor that is a posterior sequence
    :return: seq_decoded: 1-D (T) tensor that is the decoded sequence
    """
    # if not torch.is_tensor(self.outputs):
    #     warnings.warn("Outputs collected in the meter has not been processed. Call "
    #                   "MeterKWS.process_data() before calling any decoder.", RuntimeWarning)
    # else:
    # print("self.outputs", self.outputs.size())
    # self.outputs = torch.cat([F.softmax(batch, dim=-1).mean(dim=1) for batch in self.outputs], dim=0).float()

    idx_pred = np.argmax(post_seq, axis=-1)
    score_pred = np.max(post_seq, axis=-1)
    return score_pred, idx_pred


class Meter:
    def __init__(self, dict_meter_args, **kwargs):
        self.outputs = []
        self.feature_lengths = []
        self.targets = []
        self.target_lengths = []
        self.flags = []
        # self.qa_fc_final = kwargs['qa_fc_final']
        # self.aqi_fc_final = kwargs['aqi_fc_final']
        # self.aqf_fc_final = kwargs['aqf_fc_final']
        # self.y_confidence = None

        # Parameters
        self.num_classes = dict_meter_args['n_classes']
        self.idx_silence = dict_meter_args['idx_silence']
        self.smooth = dict_meter_args['smooth']
        self.smooth_window_size = dict_meter_args['smooth_window_size']
        self.confidence_window_size = dict_meter_args['confidence_window_size']
        self.zero_padding = dict_meter_args['zero_padding']
        self.threshold = dict_meter_args['threshold']

    def set_smooth_window_size(self, smooth_window_size):
        self.smooth_window_size = smooth_window_size

    def set_confidence_window_size(self, confidence_window_size):
        self.confidence_window_size = confidence_window_size

    def set_zero_padding(self, zero_padding):
        self.zero_padding = zero_padding

    def extend_data(self, outputs, feature_lengths, targets, target_lengths, flags):
        """
        :param outputs: 3-D (N, T, C) tensor that is the output of the network
        :param feature_lengths: 1-D (N) tensor that is the output of the network
        :param targets: 1-D (N) tensor that is the target vector
        :param target_lengths: 1-D (N) tensor that is the target vector
        :param flags: 1-D (N) tensor that is the flag of sample
        """
        self.outputs.extend(outputs)
        self.feature_lengths.extend(feature_lengths)
        self.targets.extend(targets)
        self.target_lengths.extend(target_lengths)
        self.flags.extend(flags)
        # # Reshape to 2-D (N*T, C)
        # temp = y_pred.reshape((-1, y_pred.size(-1))).float()
        # self.y_pred = torch.cat((self.y_pred, temp), dim=0)

    def extend_stream_data(self, outputs, targets):
        """
        :param outputs: 3-D (N, T, C) tensor that is the output of the network
        :param feature_lengths: 1-D (N) tensor that is the output of the network
        :param targets: 1-D (N) tensor that is the target vector
        :param target_lengths: 1-D (N) tensor that is the target vector
        :param flags: 1-D (N) tensor that is the flag of sample
        """
        self.outputs.extend(outputs)
        self.targets.extend(targets)

    def clear_data(self):
        # Clear Data Buffers
        self.outputs = []
        self.feature_lengths = []
        self.targets = []
        self.target_lengths = []
        self.flags = []

    def get_real_sequence(self):
        real_seq = []
        real_seq_flag = []
        for i, batch in enumerate(self.outputs):
            for j, seq in enumerate(batch):
                if self.zero_padding == 'tail':
                    real_seq.append(seq[:self.feature_lengths[i][j], :])
                else:
                    real_seq.append(seq[-self.feature_lengths[i][j]:, :])
                real_seq_flag.append(self.flags[i][j].repeat(self.feature_lengths[i][j]))
        real_seq = torch.cat(real_seq, dim=0).float().numpy()
        real_seq_flag = torch.cat(real_seq_flag, dim=0).long().numpy()
        return real_seq, real_seq_flag

    def get_decisions(self):
        """
        Greedy decoder with threshold
        :param y_pred: 2-D (N*T, C) tensor that is the output of the network
        :param threshold: Threshold for having a non-silent classification
        :return:
        """
        # Get Real Sequence
        real_seq, real_seq_flag = self.get_real_sequence()
        # real_seq = get_smoothed_seq(real_seq, self.smooth_window_size)

        # Get decision for each sample
        y_pred = []
        y_idx = []
        y_true = []
        y_confidence = []
        feature_lengths = torch.cat(self.feature_lengths, axis=0).float().numpy()
        sample_idx_head = np.cumsum(np.concatenate((np.zeros(1), feature_lengths[:-2])), axis=0).astype(np.int64)
        sample_idx_tail = np.cumsum(feature_lengths[:-1], axis=0).astype(np.int64)
        for head, tail in tqdm(zip(sample_idx_head, sample_idx_tail),
                               total=sample_idx_head.size,
                               desc='Score',
                               unit='samples'):
            # Get Current Sample
            sample_seq = real_seq[head:tail]
            sample_label = real_seq_flag[head:tail]
            sample_flag = real_seq_flag[head]
            # print(sample_flag)
            # Get Score Sequence
            if self.smooth:
                sample_seq = get_smoothed_seq(sample_seq, self.smooth_window_size)

            # Get Confidence Sequence
            # confidence_seq = get_confidence_seq(sample_seq, self.confidence_window_size)
            # max_confidence, _ = torch.max(confidence_seq, dim=0)
            # print(max_confidence)

            # Get Confidence Mask
            # seq_mask = torch.masked_fill(confidence_seq, confidence_seq < self.threshold, 0)
            # seq_mask = seq_mask.masked_fill_(seq_mask > 0, 1)

            # Get Decoded Network Output
            score_pred, seq_decoded = greedy_decoder(sample_seq)

            # Fire Control
            fire_threshold = 0
            sample_decision = np.where(score_pred > fire_threshold, seq_decoded, 0)

            y_pred_sample = sample_decision
            y_pred_sample_nodup = []

            # Remove duplications
            for i, pred_t in enumerate(y_pred_sample):
                if len(y_pred_sample_nodup) == 0 or pred_t != y_pred_sample_nodup[-1]:
                    y_pred_sample_nodup.append(pred_t)
                    # y_true_sample_nodup.append(y_true_sample[i])
            y_pred_sample_nodup = np.squeeze(np.stack(y_pred_sample_nodup).reshape(-1, 1))
            if y_pred_sample_nodup.ndim == 0:
                y_pred_sample_nodup = y_pred_sample_nodup.reshape(1)

            y_pred.append(y_pred_sample_nodup[-1])
            y_true.append(sample_flag)

        y_pred = np.squeeze(np.stack(y_pred))
        y_true = np.squeeze(np.stack(y_true))

        df = pd.DataFrame()
        df['y_true'] = pd.Series(y_true)
        df['y_pred'] = pd.Series(y_pred)
        df = df.sort_values(by=['y_true', 'y_pred'])
        df.to_csv('./prediction.csv', index=False)
        return y_true, y_pred

    def get_decisions_stream(self):
        """
        Greedy decoder with threshold
        :param y_pred: 2-D (N*T, C) tensor that is the output of the network
        :param threshold: Threshold for having a non-silent classification
        :return:
        """
        # Get Real Sequence
        score = torch.cat(self.outputs).float().numpy()
        label = torch.cat(self.targets).long().numpy()
        seq_len = label.shape[0]
        flag = [0]
        for i in range(1, seq_len):
            if label[i-1] != 0 and label[i] == 0:
                flag.append(label[i-1])
            else:
                flag.append(0)
        flag = np.asarray(flag).astype(np.int64)

        if self.smooth:
            score = get_smoothed_seq(score, self.smooth_window_size)

        # Get Decoded Network Output
        score_greedy, pred = greedy_decoder(score)

        # Firing Control
        fire_threshold = 0.1
        score = np.where(score > fire_threshold, score, 0)

        y_pred = []
        y_true = []
        for t in tqdm(range(self.smooth_window_size, flag.shape[0])):
            flag_curr = flag[t]
            score_curr = score[t]
            pred_curr = pred[t]
            score_window = score[t-self.smooth_window_size:t+1]
            pred_window = pred[t-self.smooth_window_size:t+1]

            # Check Keywords
            if flag_curr > 0:
                true = label_binarize([flag_curr], classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
                y_true.append(true)
                loc_max_score = np.argmax(score_window)
                print(true)
            #     if flag_curr in pred_window:
            #         print("True Alarm")
            #     elif pred_window[pred_window != 0].size == 0 or :
            #         print("False Rejection")

            # fire = 1 if np.max(label_window) > 0 else 0

            # Check False Rejection


            # Correct
            # nz_label = label_window[label_window != 0]
            if pred_curr > 0:
                print(pred_curr)


        y_pred = np.squeeze(np.stack(y_pred))
        y_true = np.squeeze(np.stack(y_true))

        df = pd.DataFrame()
        df['y_true'] = pd.Series(y_true)
        df['y_pred'] = pd.Series(y_pred)
        df = df.sort_values(by=['y_true', 'y_pred'])
        df.to_csv('./prediction.csv', index=False)
        return y_true, y_pred

    def get_metrics(self, dict_stat, stream=False):
        # Get Decision Sequence
        if stream:
            y_true, y_pred = self.get_decisions_stream()
        else:
            y_true, y_pred = self.get_decisions()

        # Get Confusion Matrix
        cnf_matrix = confusion_matrix(y_true, y_pred, normalize='pred')
        # print(cnf_matrix)

        FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
        FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
        TP = np.diag(cnf_matrix)
        TN = cnf_matrix.sum() - (FP + FN + TP)

        FP = FP.astype(float)
        FN = FN.astype(float)
        TP = TP.astype(float)
        TN = TN.astype(float)

        # Remove the silent label
        # if len(FP) != self.num_classes - 1:
        #     FP = FP[1:]
        # if len(FN) != self.num_classes - 1:
        #     FN = FN[1:]
        # if len(TP) != self.num_classes - 1:
        #     TP = TP[1:]
        # if len(TN) != self.num_classes - 1:
        #     TN = TN[1:]

        # Sensitivity, hit rate, recall, or true positive rate
        dict_stat['tpr'] = TP / (TP + FN)
        print(dict_stat['tpr'])
        # False negative rate
        dict_stat['fnr'] = FN / (TP + FN)
        # print(dict_stat)
        # Specificity or true negative rate
        dict_stat['tnr'] = TN / (TN + FP)
        # Precision or positive predictive value
        dict_stat['ppv'] = TP / (TP + FP)
        # Negative predictive value
        dict_stat['npv'] = TN / (TN + FN)
        # Fall out or false positive rate
        dict_stat['fpr'] = FP / (FP + TN)

        # False discovery rate
        dict_stat['fdr'] = FP / (TP + FP)
        # Overall accuracy
        dict_stat['acc'] = (TP + TN) / (TP + FP + FN + TN)
        # Micro F1 Score
        dict_stat['f1_score_micro'] = f1_score(y_true, y_pred, average='micro')
        dict_stat['cnf_matrix'] = cnf_matrix
        # Clear Data Buffers
        self.outputs = []
        self.feature_lengths = []
        self.targets = []
        self.target_lengths = []
        self.flags = []

        return dict_stat
