import torch
import torch.nn as nn
import utils.util as util


def forward_propagation(net, dict_batch_tensor, state=None):
    outputs, state, reg = net(dict_batch_tensor['features'])
    out_final, _ = outputs
    return out_final, state, reg


def get_batch_data(args, dict_batch_array):
    # Fetch Batch Data
    features = torch.from_numpy(dict_batch_array['features']).float().transpose(0, 1)  # Dim: (T, N, F)
    feature_lengths = torch.from_numpy(dict_batch_array['feature_lengths']).int()
    targets = torch.from_numpy(dict_batch_array['targets']).long()  # Training labels
    targets_metric = targets  # Labels for scoring
    target_lengths = torch.from_numpy(dict_batch_array['target_lengths']).long()
    flag = torch.from_numpy(dict_batch_array['flag']).long()
    if args.use_cuda:
        features = features.cuda()
        targets = targets.cuda()

    dict_batch_tensor = {'features': features, 'feature_lengths': feature_lengths, 'targets': targets,
                         'targets_metric': targets_metric, 'target_lengths': target_lengths, 'flag': flag}
    return dict_batch_tensor


def calculate_loss(loss_fn, net_out, dict_targets, reg, beta):
    selected_net_out = net_out
    selected_targets = dict_targets['targets']
    loss_fn_input = selected_net_out.reshape((-1, selected_net_out.size(-1)))
    loss_fn_target = selected_targets.reshape((-1))
    loss = loss_fn(loss_fn_input, loss_fn_target)
    if reg is not None:
        loss_reg = reg * beta
        loss += loss_reg
    else:
        loss_reg = 0
    return loss, loss_reg


def add_meter_data(args, meter, dict_meter_data):
    meter.extend_data(outputs=dict_meter_data['net_qout'],
                      feature_lengths=dict_meter_data['feature_lengths'],
                      targets=dict_meter_data['targets_metric'],
                      target_lengths=dict_meter_data['target_lengths'],
                      flags=dict_meter_data['flag'])
    return meter


def process_network(args, config, net, stat, alpha, epoch):
    import pickle
    net_for_eval = pickle.loads(pickle.dumps(net))

    # Process Weights for Training
    for name, param in net.named_parameters():
        # Quantize Network
        param.data = util.quantize_tensor(param.data, config['model']['wqi'], config['model']['wqf'], config['model']['qw'])
        # Column-Balanced Dropout
        if args.cbwdrop:
            if 'fc_extra' in name:
                if 'weight' in name:
                    util.cbwdrop(param.data, gamma=args.gamma_fc, alpha=alpha)
            if 'cla' in name:
                if 'weight' in name:
                    util.cbwdrop(param.data, gamma=args.gamma_rnn, alpha=alpha)
                    print("::: %s pruned by CBTD.", name)
    net_for_train = net

    # Process Weights for Evaluation
    for name, param in net_for_eval.named_parameters():
        if 'fc_final' in name:
            if stat['net_out_abs_max'] > stat['drange_max']:
                param.data = param.data / (stat['fc_final_w_scale'])
                print("###Scaling down FC Final for evaluation...")

        # Quantize Network
        param.data = util.quantize_tensor(param.data, config['model']['wqi'], config['model']['wqf'], config['model']['qw'])
        # Column-Balanced Dropout
        if args.cbwdrop:
            if 'fc_extra' in name:
                if 'weight' in name:
                    util.cbwdrop(param.data, gamma=args.gamma_fc, alpha=alpha)
            if 'cla' in name:
                if 'weight' in name:
                    util.cbwdrop(param.data, gamma=args.gamma_rnn, alpha=alpha)

    return net_for_train, net_for_eval


def quantize_network(args, net, alpha):
    for name, param in net.named_parameters():
        # Quantize Network
        param.data = util.quantize_tensor(param.data, args.wqi, args.wqf, args.qw)
        # Column-Balanced Dropout
        if args.cbwdrop:
            if 'fc_extra' in name:
                if 'weight' in name:
                    util.aligned_cbwdrop(param.data, gamma=args.gamma_fc, alpha=alpha, num_pe=args.num_array_pe)
            if 'rnn' in name:
                if 'weight' in name:
                    util.aligned_cbwdrop(param.data, gamma=args.gamma_rnn, alpha=alpha, num_pe=args.num_array_pe)
    return net


def get_net_out_stat(args, stat: dict, meter_data: dict):
    # Get max(abs(x)) of network outputs
    net_out_ravel = []
    for batch in meter_data['net_out']:
        net_out_ravel.append(batch.view(-1))
    net_out_ravel = torch.cat(net_out_ravel)
    stat['net_out_max'] = torch.max(net_out_ravel).item()
    stat['net_out_min'] = torch.min(net_out_ravel).item()
    stat['net_out_abs_max'] = torch.max(torch.abs(net_out_ravel)).item()

    # Get max(abs(x)) of network quantized outputs
    net_qout_ravel = []
    for batch in meter_data['net_qout']:
        net_qout_ravel.append(batch.view(-1))
    net_qout_ravel = torch.cat(net_qout_ravel)
    stat['net_qout_max'] = torch.max(net_qout_ravel).item()
    stat['net_qout_min'] = torch.min(net_qout_ravel).item()
    stat['net_qout_abs_max'] = torch.max(torch.abs(net_qout_ravel)).item()

    # Get dynamic range of final layer quantization
    stat['drange_max'] = float(2 ** (args.aqi_fc_final + args.aqi_fc_final - 1) - 1) / float(
        2 ** args.aqi_fc_final)
    # drange_min = -float(2 ** (args.aqi_fc_final + args.aqi_fc_final - 1)) / float(
    #     2 ** args.aqi_fc_final)
    # qstep = 1 / float(2 ** args.aqi_fc_final)

    # Get final FC layer weight scale factor
    if stat['net_out_abs_max'] > stat['drange_max']:
        stat['fc_final_w_scale'] = stat['net_out_abs_max'] / stat['drange_max']
    return stat


def initialize_network(net, args, config):
    print('::: Initializing Parameters:')
    hid_size = config['model']['hid_size']
    hid_type = config['model']['hid_type']
    fc_type = config['model']['fc_type']
    for name, param in net.named_parameters():
        print(name)
        # qLSTM uses its own initializer including quantization
        if 'cla' in name and hid_type not in ['qLSTM']:
            num_gates = int(param.shape[0] / hid_size)
            if 'bias' in name:
                nn.init.constant_(param, 0)
            if 'weight' in name:
                for i in range(0, num_gates):
                    nn.init.orthogonal_(param[i*hid_size:(i+1)*hid_size, :])
                # nn.init.xavier_normal_(param)
            if 'weight_ih_l0' in name:
                for i in range(0, num_gates):
                    nn.init.xavier_uniform_(param[i*hid_size:(i+1)*hid_size, :])

        # qLinear uses its own initializer including quantization
        if 'fc' in name and fc_type not in ['qFC']:
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            if 'bias' in name:
                # nn.init.uniform_(param)
                nn.init.constant_(param, 0)
        # if 'bias' in name:  # all biases
        #     nn.init.constant_(param, 0)
        if config['model']['hid_type'] == 'LSTM':  # only LSTM biases
            if ('bias_ih' in name) or ('bias_hh' in name):
                no4 = int(len(param) / 4)
                no2 = int(len(param) / 2)
                nn.init.constant_(param, 0)
                nn.init.constant_(param[no4:no2], 1)
    print("--------------------------------------------------------------------")
