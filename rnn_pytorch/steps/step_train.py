import random as rnd
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from nnmodels import model as model
from utils import pandaslogger, util

from tqdm import tqdm
import importlib
# from warpctc_pytorch import CTCLoss
from torch.nn import CTCLoss
from utils.util import count_net_params

old_repr = torch.Tensor.__repr__


def tensor_info(tensor):
    return repr(tensor.shape)[6:] + ' ' + repr(tensor.dtype)[6:] + '@' + str(tensor.device) + '\n' + old_repr(tensor)


def load_train_modules(dataset_name):
    # Load modules according to dataset_name
    try:
        module_log = importlib.import_module('modules.log')
    except:
        raise RuntimeError('Please select a supported dataset.')
    try:
        module_dataloader = importlib.import_module('modules.dataloader')
    except:
        raise RuntimeError('Please select a supported dataset.')
    try:
        module_train_func = importlib.import_module('modules.train_func')
    except:
        raise RuntimeError('Please select a supported dataset.')
    try:
        module_metric = importlib.import_module('modules.metric')
    except:
        raise RuntimeError('Please select a supported dataset.')
    return module_log, module_dataloader, module_train_func, module_metric


def main(args, retrain, config, **kwargs):
    torch.Tensor.__repr__ = tensor_info
    ###########################################################################################################
    # Overhead
    ###########################################################################################################
    args.retrain = retrain
    grad_history = []
    # Load Modules
    my_log, module_dataloader, module_train_func, module_metric = load_train_modules(args.dataset_name)

    # Assign methods to be used
    gen_model_id = my_log.gen_model_id
    save_best_model = my_log.save_best_model
    print_log = my_log.print_log
    gen_paths = my_log.gen_paths
    gen_log_stat = my_log.gen_log_stat
    CustomDataLoader = module_dataloader.CustomDataLoader
    process_network = module_train_func.process_network
    initialize_network = module_train_func.initialize_network
    # process_meter_data = module_metric.process_meter_data
    Meter = module_metric.Meter
    gen_meter_args = module_metric.gen_meter_args

    # Select Loss function
    dict_loss = {'crossentropy': nn.CrossEntropyLoss(reduction='mean'),
                 'ctc': CTCLoss(blank=0, reduction='sum', zero_infinity=True),
                 'mse': nn.MSELoss(), 'l1': nn.L1Loss()}
    # dict_loss = {'crossentropy': nn.CrossEntropyLoss(), 'ctc': CTCLoss(), 'mse': nn.MSELoss(), 'l1': nn.L1Loss()}
    loss_func_name = args.loss
    try:
        criterion = dict_loss[loss_func_name]
    except AttributeError:
        raise AttributeError('Please use a valid loss function. See modules/argument.py.')

    # Show Train or Retrain
    print("::: Retrain: ", retrain)
    # Show Dataset
    print("::: Loading: ", args.trainfile)
    print("::: Loading: ", args.valfile)
    print("::: Loading: ", args.testfile)

    # Find Available GPUs
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_device)
        idx_gpu = torch.cuda.current_device()
        name_gpu = torch.cuda.get_device_name(idx_gpu)
        print("::: Available GPUs: %s" % (torch.cuda.device_count()))
        print("::: Using GPU %s:   %s" % (idx_gpu, name_gpu))
        print("--------------------------------------------------------------------")
    else:
        print("::: Available GPUs: None")
        print("--------------------------------------------------------------------")

    # Reproducibility
    rnd.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # torch.set_deterministic(True)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()

    ###########################################################################################################
    # Dataloader
    ###########################################################################################################
    dataloader = CustomDataLoader(args=args, config=config)
    n_features = dataloader.n_features
    n_classes = dataloader.n_classes
    if loss_func_name == 'ctc':
        n_classes += 1  # CTC Label Shift

    ###########################################################################################################
    # Network Settings
    ###########################################################################################################
    config['model']['input_size'] = n_features
    config['model']['n_classes'] = n_classes
    # Instantiate Model
    net = model.Model(args=args, config=config)

    # Get parameter count
    n_param = count_net_params(net)
    print("::: Number of Parameters: ", n_param)

    ###########################################################################################################
    # Save & Log Naming Convention
    ###########################################################################################################

    # Generate Paths & Create Folders
    dir_paths, file_paths, pretrain_file = gen_paths(args=args, config=config, n_features=n_features,
                                                     n_classes=n_classes)
    save_dir, log_dir_hist, log_dir_best, _ = dir_paths
    save_file, logfile_hist, logfile_best, _ = file_paths
    util.create_folder([save_dir, log_dir_hist, log_dir_best])
    print("::: Save Path: ", save_file)
    print("::: Log Path: ", logfile_hist)

    # Logger
    logger = pandaslogger.PandasLogger(logfile_hist)

    ###########################################################################################################
    # Settings
    ###########################################################################################################

    # Load Pretrained Model if Running Retrain
    if retrain:
        if args.pretrain_model is None:
            print('::: Loading pretrained model: ', pretrain_file)
            net = util.load_model(args, net, pretrain_file)

        else:
            print('::: Loading pretrained model: ', args.pretrain_model)
            net = util.load_model(args, net, args.pretrain_model)

    # Use CUDA
    if args.use_cuda:
        net = net.cuda()

    # Create Optimizer
    optimizer = create_optimizer(args, net)

    # Setup Learning Rate Scheduler
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                        mode='min',
                                                        factor=args.decay_factor,
                                                        patience=args.patience,
                                                        verbose=True,
                                                        threshold=1e-4,
                                                        min_lr=args.lr_end)

    # Initialize Network Parameters
    if not retrain:
        initialize_network(net, args, config)

    # Create Meters
    dict_meter_args = gen_meter_args(args, n_classes)
    meter = Meter(dict_meter_args)

    ###########################################################################################################
    # Training
    ###########################################################################################################
    # Value for Saving Best Model
    best_metric = None
    # Timer
    start_time = time.time()

    # Epoch loop
    print("Starting training...")
    for epoch in range(args.n_epochs):
        # Update shuffle type
        # train_shuffle_type = 'random' if epoch > 100 else 'high_throughput'
        # Update Alpha
        alpha = 1 if retrain else min(epoch / (args.alpha_anneal_epoch - 1), 1.0)

        # -----------
        # Train
        # -----------
        net, _, train_stat = net_train(args,
                                       config=config,
                                       net=net,
                                       batch_size=args.batch_size,
                                       meter=None,
                                       optimizer=optimizer,
                                       criterion=criterion,
                                       dataloader=dataloader,
                                       epoch=epoch,
                                       shuffle_type='random',
                                       enable_gauss=0,
                                       grad_history=grad_history)

        # Process Network after training per epoch
        net, net_for_eval = process_network(args, config, net=net, stat=train_stat, alpha=alpha, epoch=epoch)

        # -----------
        # Validation
        # -----------
        val_stat = None
        if args.eval_val:
            _, meter, val_stat = net_eval(args,
                                          config,
                                          net=net_for_eval,
                                          set_name='val',
                                          batch_size=args.batch_size_eval,
                                          meter=meter,
                                          criterion=criterion,
                                          iterator=dataloader,
                                          epoch=epoch,
                                          shuffle_type=None,
                                          enable_gauss=0)
            if args.score_val:
                val_stat = meter.get_metrics(val_stat)
            meter.clear_data()

        # -----------
        # Test
        # -----------
        test_stat = None
        if args.eval_test:
            _, meter, test_stat = net_eval(args,
                                           config,
                                           net=net_for_eval,
                                           set_name='test',
                                           batch_size=args.batch_size_eval,
                                           meter=meter,
                                           criterion=criterion,
                                           iterator=dataloader,
                                           epoch=epoch,
                                           shuffle_type=None,
                                           enable_gauss=0)
            if args.score_test:
                test_stat = meter.get_metrics(test_stat)
            meter.clear_data()
            # print("Max: %3.4f | Min: %3.4f" % (test_stat['net_out_max'], test_stat['net_out_min']))

        ###########################################################################################################
        # Logging & Saving
        ###########################################################################################################
        # Generate Log Dict
        log_stat = gen_log_stat(args, config, epoch, start_time, retrain, net, loss_func_name, alpha,
                                optimizer, train_stat, val_stat, test_stat)

        # Write Log
        logger.load_log(log_stat=log_stat)
        logger.write_log(append=True)

        # Print
        print_log(args, log_stat, train_stat, val_stat, test_stat)

        # Save best model
        best_metric = save_best_model(args=args,
                                      best_metric=best_metric,
                                      net=net_for_eval,
                                      save_file=save_file,
                                      logger=logger,
                                      logfile_best=logfile_best,
                                      epoch=epoch,
                                      log_stat=log_stat,
                                      train_stat=train_stat,
                                      val_stat=val_stat,
                                      score_val=args.score_val)

        ###########################################################################################################
        # Learning Rate Schedule
        ###########################################################################################################
        # Schedule at the beginning of retrain
        if args.lr_schedule:
            if retrain:
                lr_scheduler.step(val_stat['lr_criterion'])
            # Schedule after the alpha annealing is over
            elif args.cbwdrop:
                if epoch >= args.alpha_anneal_epoch:
                    lr_scheduler.step(val_stat['lr_criterion'])
            else:
                lr_scheduler.step(val_stat['lr_criterion'])

    print("Training Completed...                                               ")
    print(" ")


def create_optimizer(args, net):
    if args.opt == 'ADAM':
        optimizer = optim.Adam(net.parameters(), lr=args.lr, amsgrad=False, weight_decay=args.weight_decay)
    elif args.opt == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)
    elif args.opt == 'RMSPROP':
        optimizer = optim.RMSprop(net.parameters(), lr=0.0016, alpha=0.95, eps=1e-08, weight_decay=0, momentum=0,
                                  centered=False)
    elif args.opt == 'ADAMW':
        optimizer = optim.AdamW(net.parameters(), lr=args.lr, amsgrad=False, weight_decay=args.weight_decay)
    elif args.opt == 'AdaBound':
        import adabound
        optimizer = adabound.AdaBound(net.parameters(), lr=args.lr, final_lr=0.1)
    else:
        raise RuntimeError('Please use a valid optimizer.')
    return optimizer


def plot_feature(x):
    import matplotlib.pyplot as plt
    title = ''
    index = 0
    fig, axes = plt.subplots(1, 1, figsize=(12, 6))
    im1 = axes.imshow(x.T, aspect='auto')
    plt.colorbar(im1)
    # title = "Index: %6d | Label: %2d | Keyword: %s" % (index, label, keyword)
    axes.tick_params(labelsize=16)
    axes.set_xlabel('Frames', fontsize=18)
    axes.set_ylabel('Channels', fontsize=18)
    axes.set_title(title, fontsize=18)
    # axes.set_xticks(np.arange(0, 101, 1))
    fig.savefig('./recording_plot/' + str(index) + '.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    # plt.show()


def net_eval(args,
             config,
             net,
             set_name,
             batch_size,
             meter,
             criterion,
             iterator,
             epoch,
             shuffle_type,
             enable_gauss):
    # Load modules according to dataset_name
    try:
        module_train_func = importlib.import_module('modules.train_func')
    except FileNotFoundError:
        raise RuntimeError('Please select a supported dataset.')

    # Assign methods to be used
    get_batch_data = module_train_func.get_batch_data
    calculate_loss = module_train_func.calculate_loss
    add_meter_data = module_train_func.add_meter_data
    forward_propagation = module_train_func.forward_propagation
    try:
        get_net_out_stat = module_train_func.get_net_out_stat
    except (NameError, AttributeError):
        get_net_out_stat = None
        pass

    with torch.no_grad():
        # Set Network Properties
        net = net.eval()
        net.set_qa_fc_final(config['model']['qa_fc_final'])

        # Statistics
        epoch_loss = 0.
        epoch_regularizer = 0.
        n_batches = 0

        # Meter data buffer
        dict_meter_data = {'net_out': [], 'net_qout': []}

        # Batch Iteration
        for dict_batch_array in tqdm(iterator.iterate(epoch=epoch,
                                                      set_name=set_name,
                                                      batch_size=batch_size,
                                                      mode='batch',
                                                      shuffle_type=shuffle_type,
                                                      enable_gauss=enable_gauss),
                                     desc='Eval',
                                     unit='batches',
                                     total=iterator.get_num_batch(set_name, batch_size)):

            # Get Batch Data
            dict_batch_tensor = get_batch_data(args, dict_batch_array)

            # Forward Propagation
            net_out, _, reg = forward_propagation(net, dict_batch_tensor)

            # Calculate Loss
            loss, loss_reg = calculate_loss(loss_fn=criterion,
                                            net_out=net_out,
                                            dict_targets=dict_batch_tensor,
                                            reg=reg,
                                            beta=args.beta)

            # Increment monitoring variables
            batch_loss = loss.item()
            epoch_loss += batch_loss  # Accumulate loss
            if reg:
                epoch_regularizer += loss_reg.detach().item()
            n_batches += 1  # Accumulate count so we can calculate mean later

            # Collect Meter Data
            net_out_cpu = net_out.detach().cpu()
            net_qout_cpu = util.quantize_tensor(net_out_cpu,
                                                config['model']['aqi_fc_final'],
                                                config['model']['aqf_fc_final'],
                                                1)
            dict_meter_data['net_out'].append(net_out_cpu)
            dict_meter_data['net_qout'].append(net_qout_cpu)
            for k, v in dict_batch_tensor.items():
                if k == 'features':
                    continue
                try:
                    dict_meter_data[k].append(v.detach().cpu())
                except KeyError:
                    dict_meter_data[k] = []
                    dict_meter_data[k].append(v.detach().cpu())

            # Garbage collection to free VRAM
            del dict_batch_tensor, dict_batch_array, loss, net_out

        # Average loss and regularizer values across all batches
        epoch_loss = epoch_loss / float(n_batches)
        epoch_regularizer = epoch_regularizer / float(n_batches)

        # Add meter data
        if meter is not None:
            meter = add_meter_data(args, meter, dict_meter_data)

        #######################
        # Save Statistics
        #######################
        # Add basic stats
        stat = {'loss': epoch_loss, 'reg': epoch_regularizer, 'lr_criterion': epoch_loss}
        # Get DeltaRNN Stats
        if "Delta" in config['model']['hid_type'] and args.drnn_stats:
            # Evaluate temporal sparsity
            dict_stats = net.cla.get_temporal_sparsity()
            stat['sp_dx'] = dict_stats['sparsity_delta_x']
            stat['sp_dh'] = dict_stats['sparsity_delta_h']

            # Evaluate workload
            dict_stats = net.cla.get_workload()
            print("worst_array_work: ", dict_stats['expect_worst_array_work'])
            print("mean_array_work:  ", dict_stats['expect_mean_array_work'])
            print("balance:          ", dict_stats['balance'])
            print("eff_throughput:   ", dict_stats['eff_throughput'])
            print("utilization:      ", dict_stats['utilization'])

            # net.rnn.reset_stats()
            # net.rnn.reset_debug()

        # Evaluate network output
        if get_net_out_stat is not None:
            stat = get_net_out_stat(args, stat, dict_meter_data)
        return net, meter, stat


def net_eval_stream(args,
                    config,
                    net,
                    meter,
                    criterion,
                    iterator):
    # Load modules according to dataset_name
    try:
        module_train_func = importlib.import_module('modules.' + args.dataset_name + '.train_func')
    except FileNotFoundError:
        raise RuntimeError('Please select a supported dataset.')

    # Assign methods to be used
    get_batch_data = module_train_func.get_batch_data
    calculate_loss = module_train_func.calculate_loss
    add_meter_data = module_train_func.add_meter_data
    forward_propagation = module_train_func.forward_propagation
    try:
        get_net_out_stat = module_train_func.get_net_out_stat
    except (NameError, AttributeError):
        get_net_out_stat = None
        pass

    with torch.no_grad():
        # Set Network Properties
        net = net.eval()
        net.set_qa_fc_final(config['model']['qa_fc_final'])

        # Statistics
        epoch_loss = 0.
        epoch_regularizer = 0.
        n_batches = 0

        # Meter data buffer
        dict_meter_data = {'net_out': [], 'net_qout': []}

        # Batch Iteration
        dict_batch_array = iterator.stream()

        # Get Batch Data
        # Fetch Batch Data
        features = torch.from_numpy(dict_batch_array['features']).float().unsqueeze(1)  # Dim: (T, N, F)
        targets = torch.from_numpy(dict_batch_array['targets']).long()  # Training labels
        if args.use_cuda:
            features = features.cuda()
            targets = targets.cuda()

        dict_batch_tensor = {'features': features, 'targets': targets}

        # Forward Propagation
        net_out, _, reg = forward_propagation(net, dict_batch_tensor)
        net_out = torch.squeeze(net_out)
        # Collect Meter Data
        net_out_cpu = net_out.detach().cpu()
        net_qout_cpu = util.quantize_tensor(net_out_cpu,
                                            config['model']['aqi_fc_final'],
                                            config['model']['aqf_fc_final'],
                                            1)
        dict_meter_data['net_out'].append(net_out_cpu)
        dict_meter_data['net_qout'].append(net_qout_cpu)
        for k, v in dict_batch_tensor.items():
            if k == 'features':
                continue
            try:
                dict_meter_data[k].append(v.detach().cpu())
            except KeyError:
                dict_meter_data[k] = []
                dict_meter_data[k].append(v.detach().cpu())

        # Add meter data
        if meter is not None:
            meter.extend_stream_data(outputs=dict_meter_data['net_qout'],
                                     targets=dict_meter_data['targets'])

        #######################
        # Save Statistics
        #######################
        # Add basic stats
        stat = {}
        # Get DeltaRNN Stats
        if "Delta" in config['model']['hid_type'] and args.drnn_stats:
            # Evaluate temporal sparsity
            dict_stats = net.cla.get_temporal_sparsity()
            stat['sp_dx'] = dict_stats['sparsity_delta_x']
            stat['sp_dh'] = dict_stats['sparsity_delta_h']

        # Evaluate network output
        if get_net_out_stat is not None:
            stat = get_net_out_stat(args, stat, dict_meter_data)
        return net, meter, stat


def net_train(args,
              config,
              net,
              batch_size,
              meter,
              optimizer,
              criterion,
              dataloader,
              epoch,
              shuffle_type,
              enable_gauss,
              grad_history):
    # Load modules according to dataset_name
    try:
        module_train_func = importlib.import_module('modules.train_func')
    except:
        raise RuntimeError('Please select a supported dataset.')

    # Assign methods to be used
    get_batch_data = module_train_func.get_batch_data
    calculate_loss = module_train_func.calculate_loss
    add_meter_data = module_train_func.add_meter_data
    forward_propagation = module_train_func.forward_propagation
    try:
        get_net_out_stat = module_train_func.get_net_out_stat
    except (NameError, AttributeError):
        get_net_out_stat = None
        pass

    # Set Network Properties
    net = net.train()
    net.set_qa_fc_final(0)

    # Stat
    epoch_loss = 0
    epoch_regularizer = 0
    n_batches = 0

    # Meter data buffer
    dict_meter_data = {'net_out': [], 'net_qout': []}

    # Iterate through batches
    batch_iterator = dataloader.iterate(epoch=epoch,
                                        set_name='train',
                                        batch_size=batch_size,
                                        mode='batch',
                                        shuffle_type=shuffle_type,
                                        enable_gauss=enable_gauss)
    for dict_batch_array in tqdm(batch_iterator,
                                 desc='Train',
                                 unit='batches',
                                 total=dataloader.get_num_batch('train', batch_size)):

        # Get Batch Data
        dict_batch_tensor = get_batch_data(args, dict_batch_array)

        # Optimization
        optimizer.zero_grad()

        # Forward Propagation
        net_out, _, reg = forward_propagation(net, dict_batch_tensor)

        # Calculate Loss
        loss, loss_reg = calculate_loss(loss_fn=criterion,
                                        net_out=net_out,
                                        dict_targets=dict_batch_tensor,
                                        reg=reg,
                                        beta=args.beta)

        # Get Network Outputs Statistics
        if n_batches == 0:
            net_out_min = torch.min(net_out).item()
            net_out_max = torch.max(net_out).item()
        else:
            min_cand = torch.min(net_out)
            max_cand = torch.max(net_out)
            if min_cand < net_out_min:
                net_out_min = min_cand.item()
            if max_cand > net_out_max:
                net_out_max = max_cand.item()

        # Backward propagation
        loss.backward()

        # Gradient clipping
        if args.clip_grad_norm_max != 0:
            autoclip_gradient(net, grad_history, 10)
            # nn.utils.clip_grad_norm_(net.parameters(), args.clip_grad_norm_max)

        # Update parameters
        optimizer.step()

        # Increment monitoring variables
        loss.detach()
        batch_loss = loss.item()
        epoch_loss += batch_loss  # Accumulate loss
        if reg is not None:
            epoch_regularizer += loss_reg.detach().item()
        n_batches += 1  # Accumulate count so we can calculate mean later

        # Collect Meter Data
        net_out_cpu = net_out.detach().cpu()
        net_qout_cpu = util.quantize_tensor(net_out_cpu,
                                            config['model']['aqi_fc_final'],
                                            config['model']['aqf_fc_final'],
                                            1)
        dict_meter_data['net_out'].append(net_out_cpu)
        dict_meter_data['net_qout'].append(net_qout_cpu)
        for k, v in dict_batch_tensor.items():
            if k == 'features':
                continue
            try:
                dict_meter_data[k].append(v.detach().cpu())
            except KeyError:
                dict_meter_data[k] = []

        # Garbage collection to free VRAM
        del dict_batch_tensor, dict_batch_array, loss, reg, net_out

    # Average loss and regularizer values across batches
    epoch_loss /= n_batches
    epoch_loss = epoch_loss
    epoch_regularizer /= n_batches

    # Collect outputs and targets
    if meter is not None:
        meter = add_meter_data(meter, dict_meter_data)

    # Get network statistics
    stat = {'loss': epoch_loss, 'reg': epoch_regularizer, 'net_out_min': net_out_min, 'net_out_max': net_out_max}
    if get_net_out_stat is not None:
        stat = get_net_out_stat(args, stat, dict_meter_data)
    return net, meter, stat


def _get_grad_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm


def autoclip_gradient(model, grad_history, clip_percentile):
    """
    Cite https://github.com/pseeth/autoclip/blob/master/autoclip.pdf
    """
    obs_grad_norm = _get_grad_norm(model)
    grad_history.append(obs_grad_norm)
    clip_value = np.percentile(grad_history, clip_percentile)
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
    # print("clip_value: ", clip_value)
