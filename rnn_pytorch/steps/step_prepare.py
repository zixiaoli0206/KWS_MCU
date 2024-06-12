import os
import numpy as np
import utils.util as util
# from modules import data_collector
import importlib


def main(args, config, proj_root):
    np.random.seed(0)
    dataset_path = './data/speech_commands_v0.02'
    testset_path = './data/speech_commands_test_set_v0.02'

    # Dictionary for outputs
    output_path = os.path.join(proj_root, 'data')

    # Load modules according to dataset_name
    try:
        module_data = importlib.import_module('modules.prepare')
        DataCollector = module_data.DataPrepare
    except:
        raise RuntimeError('Please select a supported dataset.')

    # Data Augmentation
    all_speeds = [0.9, 1.1]  # speed variations
    list_snr = [10, 5]  # noise levels

    print("Preparing: ", args.dataset_name)
    prepare = DataCollector(args=args,
                            config=config,
                            dataset_path=dataset_path,
                            testset_path=testset_path,
                            output_path=output_path)

    # Creat Silence Samples
    prepare.create_silence()

    # Collect Dataset
    prepare.collect()



def gen_meter_args(args, dataset_path, output_path):
    dict_meter_args = {'dataset_path': dataset_path,
                       'output_path': output_path,
                       'n_targets': args.n_targets}
    if args.dataset_name == 'timit':
        dict_meter_args['phn'] = args.phn
    return dict_meter_args
