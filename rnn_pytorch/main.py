__author__ = "Chang Gao"
__copyright__ = "Copyright 2020"
__credits__ = ["Chang Gao", "Stefan Braun"]
__license__ = "Private"
__version__ = "0.1.0"
__maintainer__ = "Chang Gao"
__email__ = "chang.gao@uzh.ch"
__status__ = "Prototype"

import os
import utils.argument as argument
from steps import step_prepare, step_feature, step_train
import importlib
import json

if __name__ == '__main__':

    # Process Arguments
    arg_parser = argument.ArgProcessor()
    args = arg_parser.get_args()

    # Root Path of this Python Project
    path_root = os.path.dirname(os.path.abspath(__file__))

    # Load Config File
    config_path = os.path.join(path_root, 'config', args.config)
    with open(config_path) as config_file:
        config = json.load(config_file)

    # Use Default Feature File Paths if not specified
    try:
        module_log = importlib.import_module('modules.log')
    except:
        raise RuntimeError('Please select a supported dataset.')

    if args.trainfile is None or args.valfile is None or args.testfile is None:
        print("Loading from default feature file Paths...")
        _, args.trainfile, args.valfile = module_log.gen_trainset_name(args, config)
        args.testfile = module_log.gen_testset_name(args, config)

    args.trainfile = os.path.join(path_root, 'feat', args.trainfile)
    args.valfile = os.path.join(path_root, 'feat', args.valfile)
    args.testfile = os.path.join(path_root, 'feat', args.testfile)

    # Step 0 - GSCD Data Preparation
    if args.step == 'prepare':
        print("####################################################################################################")
        print("# Step 0: Data Preparation                                                                         #")
        print("####################################################################################################")
        step_prepare.main(args, config, proj_root=path_root)

    # Step 1 - Feature Extraction
    if args.step == 'feature':
        print("####################################################################################################")
        print("# Step 1: Feature Extraction                                                                       #")
        print("####################################################################################################")
        step_feature.main(args=args,
                           config=config,
                           path_root=path_root)

    # Step 2 - Pretrain
    if args.step == 'pretrain':
        print("####################################################################################################")
        print("# Step 2: Pretrain                                                                                 #")
        print("####################################################################################################")
        retrain = 0
        step_train.main(args, retrain, config)