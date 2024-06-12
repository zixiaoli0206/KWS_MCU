import os
import typing
import warnings

import numpy as np
import torch
import pandas as pd

from utils import util


def gen_paths(args,
              config,
              n_features: int,
              n_classes: int):
    model_id, pretrain_model_id = gen_model_id(args=args, config=config, n_features=n_features, n_classes=n_classes)

    dataset_name = args.dataset_name
    save_dir = os.path.join('./save', args.step)  # Best model save dir
    log_dir_hist = os.path.join('./log', args.step, 'hist')  # Log dir to save training history
    log_dir_best = os.path.join('./log', args.step, 'best')  # Log dir to save info of the best epoch
    log_dir_test = os.path.join('./log', args.step, 'test')  # Log dir to save info of the best epoch
    dir_paths = (save_dir, log_dir_hist, log_dir_best, log_dir_test)

    # File Paths
    if model_id is not None:
        logfile_hist = os.path.join(log_dir_hist, model_id + '.csv')  # .csv logfile_hist
        logfile_best = os.path.join(log_dir_best, model_id + '.csv')  # .csv logfile_hist
        logfile_test = os.path.join(log_dir_test, model_id + '.csv')  # .csv logfile_hist
        save_file = os.path.join(save_dir, model_id + '.pt')
    if model_id is not None:
        file_paths = (save_file, logfile_hist, logfile_best, logfile_test)
    else:
        file_paths = None

    # Pretrain Model Path
    if pretrain_model_id is not None:
        pretrain_file = os.path.join('./save', 'pretrain', pretrain_model_id + '.pt')
    else:
        pretrain_file = None

    return dir_paths, file_paths, pretrain_file


class Logger:
    def __init__(self, logfile):
        self.logfile = logfile
        self.list_header = []
        self.loglist = []

    def add_row(self, list_header, list_value):
        self.list_header = list_header
        row = {}
        for header, value in zip(list_header, list_value):
            row[header] = value
        self.loglist.append(row)

    def write_log(self, logfile=None):
        if len(self.list_header) == 0:
            warnings.warn("DataFrame columns not defined. Call add_row for at least once...", RuntimeWarning)
        else:
            df = pd.DataFrame(self.loglist, columns=self.list_header)
            if logfile is not None:
                df.to_csv(logfile, index=False)
            else:
                df.to_csv(self.logfile, index=False)

    def write_log_idx(self, idx, logfile=None):
        if len(self.list_header) == 0:
            warnings.warn("DataFrame columns not defined. Call add_row for at least once...", RuntimeWarning)
        else:
            loglist_best = [self.loglist[idx]]
            df = pd.DataFrame(loglist_best, columns=self.list_header)
            if logfile is not None:
                df.to_csv(logfile, index=False)
            else:
                df.to_csv(self.logfile, index=False)


def gen_trainset_name(args, config):
    # Feat name
    feat_type = config["feature"]["type"]
    dataset_id = '_D_' + args.dataset_name \
                 + '_FT_' + feat_type \
                 + '_NF_' + str(config['feature']['n_filt']) \
                 + '_SI_' + str(config['feature']['frame_size']) \
                 + '_ST_' + str(config['feature']['frame_stride']) \
                 + '_FL_' + str(config['feature']['freq_low']) \
                 + '_FH_' + str(config['feature']['freq_high']) \
                 + '_SP_' + config["feature"][feat_type]["f_spacing"] \
                 + '_DT_' + str(config["feature"]['delta_t'])

    trainfile = 'TRAIN' + dataset_id
    valfile = 'VAL' + dataset_id
    testfile = 'TEST' + dataset_id

    trainfile += '.h5'
    valfile += '.h5'
    testfile += '.h5'

    return dataset_id, trainfile, valfile


def gen_testset_name(args, config):
    # Feat name
    feat_type = config["feature"]["type"]
    dataset_id = '_D_' + args.dataset_name \
                 + '_FT_' + feat_type \
                 + '_NF_' + str(config['feature']['n_filt']) \
                 + '_SI_' + str(config['feature']['frame_size']) \
                 + '_ST_' + str(config['feature']['frame_stride']) \
                 + '_FL_' + str(config['feature']['freq_low']) \
                 + '_FH_' + str(config['feature']['freq_high']) \
                 + '_SP_' + config["feature"][feat_type]["f_spacing"] \
                 + '_DT_' + str(config["feature"]['delta_t'])
    testfile = 'TEST' + dataset_id
    testfile += '.h5'
    return testfile


def gen_model_id(args, config, **kwargs):
    # Custom String
    str_custom = '' if args.filename is None else args.filename + '_'

    # Setting Description
    str_setting = 'SD_' + f"{args.seed:d}"

    # Feature Description
    # trainset_name, _, _ = gen_trainset_name(args, config)

    str_feat = '_SNR_' + str(args.snr) \
               + '_QF_' + str(config['feature']['qf']) \
               + '_FBW_' + str(config['feature']['bw_feat']) \
               + '_LOG_' + str(config['feature']['log_feat']) \
               + '_NOR_' + str(config['feature']['normalization'])

    # Architecture Description
    str_net_arch = "_IN_" + f"{kwargs['n_features']:d}" \
                   + '_L_' + f"{config['model']['hid_layers']:d}" \
                   + '_H_' + f"{config['model']['hid_size']:d}" \
                   + '_D_' + f"{config['model']['hid_dropout']:.2f}" \
                   + '_CLA_' + config['model']['hid_type']
    # Add FC Layer
    str_net_arch += '_FC_' + f"{config['model']['fc_extra_size']:d}"

    # Add Number of Classes
    str_net_arch += '_NC_' + f"{kwargs['n_classes']:d}"

    # Quantization of Activation
    str_net_arch += '_QA_' + str(config['model']['qa']) \
                    + '_AQI_' + f"{config['model']['aqi']:d}" \
                    + '_AQF_' + f"{config['model']['aqf']:d}"
    # Quantization of Weights
    str_net_arch += '_QW_' + str(config['model']['qw']) \
                    + '_WQI_' + f"{config['model']['wqi']:d}" \
                    + '_WQF_' + f"{config['model']['wqf']:d}"
    # Pretrain Model ID
    pretrain_model_id = str_custom + str_setting + str_feat + str_net_arch
    pretrain_model_id = pretrain_model_id.replace("Delta",
                                                  "")  # Remove "Delta" from the model ID to load the non-delta network

    # Delta Network
    if 'Delta' in config['model']['hid_type']:
        str_net_arch += '_TX_' + f"{config['model']['th_x']:.2f}" \
                        + '_TH_' + f"{config['model']['th_h']:.2f}" \
                        + '_BT_' + f"{args.beta:.1e}"
        # str_net_arch += '_TX_' + f"{0:.2f}" \
        #                 + '_TH_' + f"{0:.2f}" \
        #                 + '_BT_' + f"{args.beta:.1e}"

    # Model ID
    model_id = str_custom + str_setting + str_feat + str_net_arch

    return model_id, pretrain_model_id


def write_log(args, logger, tb_writer, model_id, train_stat, val_stat, test_stat, net, optimizer, epoch, time_curr,
              alpha, retrain):
    def get_dict_keyword():
        dict_keyword2label = {}
        dict_label2keyword = {}
        label_list = pd.read_csv('./data/' + args.dataset_name + '/label_list_10_test.csv')
        for row in label_list.itertuples():
            dict_keyword2label[str(row.keyword)] = row.label
            dict_label2keyword[row.label] = {str(row.keyword)}
        return dict_keyword2label, dict_label2keyword

    # Get Dictionaries for Conversion between Keywords & Labels
    dict_keyword2label, dict_label2keyword = get_dict_keyword()

    # Get parameter count
    n_param = 0
    for name, param in net.named_parameters():
        sizes = 1
        for el in param.size():
            sizes = sizes * el
        n_param += sizes

    # Evaluate Weight Range
    # for name, param in net.named_parameters():
    #     param_data = param.data
    #     print("Name: %30s | Min: %f | Max: %f" % (name, torch.min(param_data), torch.max(param_data)))

    # Evaluate RNN Weight Sparsity
    n_nonzero_weight_elem = 0
    n_weight_elem = 0
    for name, param in net.named_parameters():
        if 'rnn' in name:
            if 'weight' in name:
                n_nonzero_weight_elem += len(param.data.nonzero())
                n_weight_elem += param.data.nelement()
    sp_w_rnn = 1 - (n_nonzero_weight_elem / n_weight_elem)

    # Evaluate FC Layer Weight Sparsity
    sp_w_fc = 0
    if args.fc_extra_size:
        n_nonzero_weight_elem = 0
        n_weight_elem = 0
        for name, param in net.named_parameters():
            if 'fc_extra' in name:
                if 'weight' in name:
                    n_nonzero_weight_elem += len(param.data.nonzero())
                    n_weight_elem += param.data.nelement()
        sp_w_fc = 1 - (n_nonzero_weight_elem / n_weight_elem)

    # Get current learning rate
    lr_curr = 0
    if optimizer is not None:
        for param_group in optimizer.param_groups:
            lr_curr = param_group['lr']

    # Create Log List
    list_log_headers = ['EPOCH', 'TIME', 'N_PARAM', 'alpha']
    list_log_values = [epoch, time_curr, n_param, alpha]
    if train_stat is not None:
        list_log_headers.append('LR')
        list_log_values.append(lr_curr)
    if train_stat is not None:
        list_log_headers.append('LOSS_TRAIN')
        list_log_values.append(train_stat['loss'])
    if val_stat is not None:
        list_log_headers.append('LOSS_VAL')
        list_log_values.append(val_stat['loss'])
        if args.score_val:
            list_log_headers.extend(['ACC-VAL', 'SP-DX', 'SP-DH'])
            list_log_values.extend([val_stat['f1_score_micro'], val_stat['sp_dx'], val_stat['sp_dh']])
    if test_stat is not None:
        list_log_headers.extend(['LOSS_TEST', 'ACC-TEST', 'SP-DX', 'SP-DH'])
        list_log_values.extend([test_stat['loss'], test_stat['f1_score_micro'], test_stat['sp_dx'], test_stat['sp_dh']])
        list_log_headers.extend([key for key in dict_keyword2label.keys()])
        list_log_values.extend(test_stat['tpr'])

    # Write Log
    logger.add_row(list_log_headers, list_log_values)
    logger.write_csv()

    # Print Info
    if retrain:
        n_epochs = args.n_epochs_retrain
    else:
        n_epochs = args.n_epochs

    str_print = f"Epoch: {epoch + 1:3d} of {n_epochs:3d} | Time: {time_curr:s} | LR: {lr_curr:1.5f} | Sp.W {sp_w_rnn * 100:3.2f}%% | Sp.Wfc {sp_w_fc * 100:3.2f}%% |\n"
    if train_stat is not None:
        str_print += f"    | Train-Loss: {train_stat['loss']:4.2f} | Train-Reg: {train_stat['reg']:4.2f} |\n"
    if val_stat is not None:
        str_print += f"    | Val-Loss: {val_stat['loss']:4.3f}"
        if args.score_val:
            str_print += f" | Val-F1: {val_stat['f1_score_micro'] * 100:3.2f} | Val-Sp-dx: {val_stat['sp_dx'] * 100:3.2f} | Val-Sp" \
                         f"-dh {val_stat['sp_dh'] * 100:3.2f} |"
        str_print += '\n'
    if test_stat is not None:
        str_print += f"    | Test-Loss: {test_stat['loss']:4.3f}"
        if args.score_test:
            str_print += f" | Test-F1: {test_stat['f1_score_micro'] * 100:3.2f} | Test-Sp-dx: {test_stat['sp_dx'] * 100:3.2f} | Test-Sp" \
                         f"-dh: {test_stat['sp_dh'] * 100:3.2f} | "
    print(str_print)

    # Tensorboard
    if tb_writer is not None:
        tb_writer.add_scalars(model_id, {'L-Train': train_stat['loss'],
                                         'L-Val': val_stat['loss']}, epoch)


def save_best_model(args, best_metric, net, save_file, logger, logfile_best, epoch, log_stat, train_stat, val_stat,
                    score_val):
    if args.save_every_epoch:
        best_metric = 0
        best_epoch = epoch
        logger.write_log(append=False, logfile=logfile_best)
        torch.save(net.state_dict(), save_file)
        print('>>> saving best model from epoch %d to %s' % (epoch, save_file))
    else:  # Save every epoch
        if score_val:
            if epoch == 0 or val_stat['f1_score_micro'] > best_metric:
                best_metric = val_stat['f1_score_micro']
                best_cnf = val_stat['cnf_matrix']
                save_file_cnf = save_file.replace('pt', 'npy')
                np.save(save_file_cnf, best_cnf)
                logger.write_log(append=False, logfile=logfile_best)
                torch.save(net.state_dict(), save_file)
                best_epoch = epoch
                print('>>> saving best model from epoch %d to %s' % (epoch, save_file))
        else:
            if epoch == 0 or val_stat['loss'] < best_metric:
                best_metric = val_stat['loss']
                torch.save(net.state_dict(), save_file)
                logger.write_log(append=False, logfile=logfile_best)
                best_epoch = epoch
                print('>>> saving best model from epoch %d to %s' % (epoch, save_file))
    print("Best Metric: ", best_metric)
    return best_metric


def print_log(args, log_stat, train_stat, val_stat, test_stat):
    str_print = f"Epoch: {log_stat['EPOCH'] + 1:3d} of {log_stat['N_EPOCH']:3d} " \
                f"| Time: {log_stat['TIME_CURR']:s} " \
                f"| LR: {log_stat['LR_CURR']:1.5f} " \
                f"| Sp.W {log_stat['SP_W_CLA'] * 100:3.2f}%% " \
                f"| Sp.Wfc {log_stat['SP_W_FC'] * 100:3.2f}%% |\n"
    if train_stat is not None:
        str_print += f"    | Train-Loss: {log_stat['TRAIN_LOSS']:4.3f} " \
                     f"| Train-Reg: {log_stat['TRAIN_REG']:4.2f} |\n"
    if val_stat is not None:
        str_print += f"    | Val-Loss: {log_stat['VAL_LOSS']:4.3f}"
        if args.score_val:
            str_print += f" | Val-ACC: {log_stat['VAL_F1_SCORE_MICRO'] * 100:3.3f}% " \
                # f"| Val-Sp-dx: {log_stat['VAL_SP_DX'] * 100:3.2f} " \
            # f"| Val-Sp-dh: {log_stat['VAL_SP_DH'] * 100:3.2f} |"
        str_print += '\n'
    if test_stat is not None:
        str_print += f"    | Test-Loss: {log_stat['TEST_LOSS']:4.3f}"
        if args.score_test:
            str_print += f" | Test-ACC: {log_stat['TEST_F1_SCORE_MICRO'] * 100:3.3f}% " \
                # f"| Test-SP-DX: {log_stat['TEST_SP_DX'] * 100:3.2f} " \
            # f"| Test-SP-DH: {log_stat['TEST_SP_DH'] * 100:3.2f} | "
    print(str_print)


def gen_log_stat(args, config, epoch, start_time, retrain, net, loss_func_name, alpha,
                 optimizer=None, train_stat=None, val_stat=None, test_stat=None):
    # End Timer
    time_curr = util.timeSince(start_time)

    # Get Epoch & Batch Size
    n_epochs = args.n_epochs
    batch_size = args.batch_size

    # Get current learning rate
    lr_curr = 0
    if optimizer is not None:
        for param_group in optimizer.param_groups:
            lr_curr = param_group['lr']

    # Get parameter count
    n_param = 0
    for name, param in net.named_parameters():
        sizes = 1
        for el in param.size():
            sizes = sizes * el
        n_param += sizes

    # Create log dictionary
    log_stat = {'SEED': args.seed,
                'EPOCH': epoch,
                'N_EPOCH': n_epochs,
                'TIME_CURR': time_curr,
                'BATCH_SIZE': batch_size,
                'N_PARAM': n_param,
                'LOSS_FUNC': loss_func_name,
                'OPT': args.opt,
                'LR_CURR': lr_curr,
                'HID_DROPOUT': config['model']['hid_dropout']
                }

    # Merge stat dicts into the log dict
    if train_stat is not None:
        train_stat_log = {f'TRAIN_{k.upper()}': v for k, v in train_stat.items()}
        log_stat = {**log_stat, **train_stat_log}
    if val_stat is not None:
        val_stat_log = {f'VAL_{k.upper()}': v for k, v in val_stat.items()}
        del val_stat_log['VAL_LR_CRITERION']
        log_stat = {**log_stat, **val_stat_log}
    if test_stat is not None:
        test_stat_log = {f'TEST_{k.upper()}': v for k, v in test_stat.items()}
        del test_stat_log['TEST_LR_CRITERION']
        log_stat = {**log_stat, **test_stat_log}

    # Evaluate Classifier Weight Sparsity
    if config['model']['hid_type'] != 'FC':
        n_nonzero_weight_elem = 0
        n_weight_elem = 0
        for name, param in net.named_parameters():
            if 'cla' in name:
                if 'weight' in name:
                    n_nonzero_weight_elem += len(torch.nonzero(param.data))
                    n_weight_elem += param.data.nelement()
        log_stat['SP_W_CLA'] = 1 - (n_nonzero_weight_elem / n_weight_elem)

    # Evaluate FC Extra Layer Weight Sparsity
    log_stat['SP_W_FC'] = 0
    if config['model']['fc_extra_size']:
        n_nonzero_weight_elem = 0
        n_weight_elem = 0
        for name, param in net.named_parameters():
            if 'fc_extra' in name:
                if 'weight' in name:
                    n_nonzero_weight_elem += len(torch.nonzero(param.data))
                    n_weight_elem += param.data.nelement()
        log_stat['SP_W_FC'] = 1 - (n_nonzero_weight_elem / n_weight_elem)
    return log_stat
