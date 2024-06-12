###########################################################################################################
# Google Speech Command Dataset Version 2
###########################################################################################################


def add_args(parser):
    # Config
    parser.add_argument('--config', default='debug.json', help='Name of config file')
    # Export
    parser.add_argument('--export_model', default=None, help='Path of the model to be exported for hardware')
    parser.add_argument('--hw_id', default='edgedrnn', help='Hardware platform')
    # Dataset
    parser.add_argument('--data_dir', default='./data/gscdv2', help='Directory path that saves datasets')
    # parser.add_argument('--augment_noise', default=0, type=int, help='0 - Not augment noise | 1 - Augment noise')
    # parser.add_argument('--train_noise', default=0, type=int, help='Train with noisy data')
    parser.add_argument('--snr', default=0, type=int, help='Signal-to-Noise ratio for test')
    parser.add_argument('--trainfile', default=None, help='HDF5 File of training set')
    parser.add_argument('--valfile', default=None, help='HDF5 File of validation set')
    parser.add_argument('--testfile', default=None, help='HDF5 File of testing set')
    parser.add_argument('--zero_padding', default='head',
                        help='Method of padding zeros to samples in a batch')
    # Feature Extraction
    parser.add_argument('--feat_name', default='', help='A string append to the dataset file names')
    parser.add_argument('--plt_feat',  default=0, type=int, help='Frame size of signals')
    parser.add_argument('--use_vad', default=0, type=int, help='Use VAD for digital feature extraction')

    # Logging
    parser.add_argument('--filename', default=None, help='Filename to save model and log to.')
    parser.add_argument('--save_every_epoch', default=0, type=int, help='Save model for every epoch.')
    # Hyperparameters
    parser.add_argument('--seed', default=0, type=int,
                        help='Initialize the random seed of the run (for reproducibility).')
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size for training.')
    parser.add_argument('--batch_size_retrain', default=256, type=int, help='Batch size for training.')
    parser.add_argument('--batch_size_eval', default=512, type=int, help='Batch size for evaluation.')
    parser.add_argument('--n_epochs', default=50, type=int, help='Number of epochs to train for.')
    parser.add_argument('--n_epochs_retrain', default=50, type=int, help='Number of epochs to train for.')
    parser.add_argument('--loss', default='crossentropy', help='Loss function.')
    parser.add_argument('--opt', default='ADAMW', help='Which optimizer to use (ADAM or SGD)')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('--lr_schedule', default=1, type=int, help='Whether enable learning rate scheduling')
    parser.add_argument('--lr_end', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--lr_retrain', default=5e-4, type=float, help='Learning rate')
    parser.add_argument('--decay_factor', default=0.8, type=float, help='Learning rate')
    parser.add_argument('--patience', default=3, type=float, help='Learning rate')
    parser.add_argument('--beta', default=0, type=float,
                        help='Regularization factor')
    parser.add_argument('--clip_grad_norm_max', default=100, type=float, help='Gradient clipping')
    parser.add_argument('--weight_decay', default=0.01, type=float, help='Weight decay factor')
    # General Network Settings
    parser.add_argument('--inp_dropout', default=0, type=float,  # 0.2
                       help='Dropout rate of the input layer')

    parser.add_argument('--hardsigmoid', default=0, type=int, help='Whether use hardsigmoid')
    parser.add_argument('--hardtanh', default=0, type=int, help='Whether use hardtanh')
    # Convolutional Network Settings
    parser.add_argument('--context_window_size', default=256, type=int, help='Context window size')
    # Delta Network Settings
    parser.add_argument('--drnn_stats', default=0, type=int, help='Whether collect DRNN stats')
    parser.add_argument('--nqi', default=2, type=int,
                        help='Number of integer bits before LUT output decimal point')
    parser.add_argument('--nqf', default=6, type=int,
                        help='Number of fraction bits after LUT output decimal point') # 5
    parser.add_argument('--qg', default=0, type=int,
                        help='Number of fraction bits after gradient decimal point')
    parser.add_argument('--gqi', default=1, type=int,
                        help='Number of integer bits before gradient decimal point')
    parser.add_argument('--gqf', default=9, type=int,
                        help='Number of fraction bits after gradient decimal point')
    parser.add_argument('--qa_fc_extra', default=0, type=int,
                        help='Number of fraction bits after activation decimal point for the extra FC layer')
    parser.add_argument('--aqi_fc_extra', default=8, type=int,
                        help='Number of integer bits before activation decimal point for the extra FC layer')
    parser.add_argument('--aqf_fc_extra', default=8, type=int,
                        help='Number of fraction bits after activation decimal point for the extra FC layer')
    parser.add_argument('--qa_fc_final', default=1, type=int,
                        help='Number of fraction bits after activation decimal point for the final FC layer')
    parser.add_argument('--aqi_fc_final', default=6, type=int,
                        help='Number of integer bits before activation decimal point for the final FC layer')
    parser.add_argument('--aqf_fc_final', default=8, type=int,
                        help='Number of fraction bits after activation decimal point for the final FC layer')
    # parser.add_argument('--th_x', default=1/256.0, type=float, help='Whether quantize the network weights')
    # parser.add_argument('--th_h', default=1/256.0, type=float, help='Whether quantize the network weights')
    # Scoring Settings
    parser.add_argument('--smooth', default=1, type=int, help='Whether smooth the posterior over time')
    parser.add_argument('--smooth_window_size', default=47, type=int, help='Posterior smooth window size')
    parser.add_argument('--confidence_window_size', default=80, type=int,
                        help='Confidence score window size')
    parser.add_argument('--fire_threshold', default=0, type=float,
                        help='Threshold fortrain (1) firing a decision')
    # Training Process
    parser.add_argument('--step', default='train', help='Which step to start from')
    parser.add_argument('--benchmark', default=0, type=int, help='Toggle benchmark mode of the model')
    parser.add_argument('--debug', default=0, type=int, help='Toggle debug mode of the model')
    parser.add_argument('--run_through', default=0, type=int, help='Whether run through rest steps')
    parser.add_argument('--run_retrain', default=0, type=int,
                        help='Whether run retrain steps after pretrain')
    parser.add_argument('--eval_val', default=1, type=int, help='Whether eval val set during training')
    parser.add_argument('--score_val', default=1, type=int, help='Whether score val set during training')
    parser.add_argument('--eval_test', default=1, type=int, help='Whether eval test set during training')
    parser.add_argument('--score_test', default=1, type=int, help='Whether score test set during training')
    parser.add_argument('--eval_sp', default=1, type=int, help='Whether run through rest steps')
    parser.add_argument('--iter_mode', default='batch', help='Dynamic batch size.')
    # parser.add_argument('--normalization', default='standard', help='Custom pretrained model')
    parser.add_argument('--pretrain_model', default=None, help='Custom pretrained model')
    parser.add_argument('--use_tensorboard', default=0, type=int, help='Custom pretrained model')
    parser.add_argument('--use_cuda', default=1, type=int, help='Use GPU yes/no')
    parser.add_argument('--gpu_device', default=0, type=int, help='Select GPU')
    # Column Balanced Targeted Dropout
    parser.add_argument('--cbwdrop', default=0, type=int,
                        help='Whether use Column-Balanced Weight Dropout')
    parser.add_argument('--gamma_rnn', default=0.3, type=float, help='Target sparsity of cbwdrop')
    parser.add_argument('--gamma_fc', default=0, type=float, help='Target sparsity of cbwdrop')
    parser.add_argument('--num_array_pe', default=1, type=int, help='Number of PEs per PE Array')
    parser.add_argument('--alpha_anneal_epoch', default=30, type=int, help='Target sparsity of cbwdrop')
    # Test
    parser.add_argument('--test_vol_db', default=0, type=int, help='Target sparsity of cbwdrop')
    return parser
