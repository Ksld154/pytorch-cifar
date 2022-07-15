import argparse


def opt_parser():
    usage = 'Trains and tests a Gradual layer freezing LeNet-5 model with CIFAR10.'
    parser = argparse.ArgumentParser(description=usage)

    parser.add_argument('-o',
                        '--overlap',
                        default=True,
                        dest='transmission_overlap',
                        help='Transmission overlap with next model training (default: %(default)s)',
                        action=argparse.BooleanOptionalAction)
    parser.add_argument('-d',
                        '--dryrun',
                        default=False,
                        dest='dry_run',
                        help='Only do pre-training (default: %(default)s)',
                        action=argparse.BooleanOptionalAction)
    parser.add_argument('-a',
                        '--all',
                        default=False,
                        dest='all_experiments',
                        help='Do 3 experiments at once (default: %(default)s)',
                        action=argparse.BooleanOptionalAction)
    parser.add_argument('-t',
                        '--transmission-time',
                        default=30,
                        type=int,
                        dest='transmission_time',
                        help='Mock tensor transmission time (default: %(default)s)')
    parser.add_argument('-e',
                        '--epochs',
                        default=10,
                        type=int,
                        help='Training epoches (default: %(default))')
    parser.add_argument('-s',
                        '--switch',
                        default=True,
                        dest='switch_model',
                        help='Enable model switching (default: %(default)s)',
                        action=argparse.BooleanOptionalAction)
    parser.add_argument('-u',
                        '--utility',
                        default=True,
                        dest='utility_flag',
                        help='Use model utility to decide switch model or not (default: %(default)s)',
                        action=argparse.BooleanOptionalAction)

    parser.add_argument('-g',
                        '--gpu',
                        default=3,
                        dest='gpu_device',
                        help='Specify which gpu device to use (default: %(default)s)',
                        type=int)
    parser.add_argument('-f',
                        '--force-switch',
                        default=0,
                        dest='force_switch_epoch',
                        type=int,
                        help='Force to switch model at specific epoch (default: %(default)s)')
    parser.add_argument('--brute-force',
                        default=False,
                        dest='list_all_candidate_models',
                        action=argparse.BooleanOptionalAction,
                        help='List and train every candidate models with different freezing degree (default: %(default)s)')
    parser.add_argument('--freeze-idx',
                        default=0,
                        dest='freeze_idx',
                        type=int,
                        help='Freeze degree (default: %(default)s)')

    ### Active Options ###
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--model',
                        default='mobilenet',
                        dest='model',
                        type=str,
                        help='Model Type (default: %(default)s)')
    
    parser.add_argument('--window_size', 
                        type=int, default=5,
                        help='Moving average window size for model loss difference (default: %(default)s)')

    parser.add_argument('--gradually-freeze',
                        default=True,
                        dest='gradually_freeze',
                        action=argparse.BooleanOptionalAction,
                        help='Train Gradually Freeze Models (default: %(default)s)')

    parser.add_argument('--static-freeze',
                        default=True,
                        dest='static_freeze',
                        action=argparse.BooleanOptionalAction,
                        help='Train all Static Freeze Models (default: %(default)s)')

    parser.add_argument('--static-freeze-candidates',
                        default=5,
                        dest='static_freeze_candidates',
                        type=int,
                        help='Candidate Static Freeze Degree (default: %(default)s)')

    
    parser.add_argument('--pre-epochs',
                        dest='pre_epochs_ratio',
                        default=0.1,
                        type=float,
                        help='Pre-Training epoches (default: %(default))')

    parser.add_argument('--loss_diff_ratio',
                        dest='loss_diff_ratio_threshold',
                        default=0.05,
                        type=float,
                        help='Threshold of loss difference deciding when to switch model (default: %(default))')

    return parser.parse_args()
