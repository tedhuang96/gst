import pathhack
import argparse
import torch
import numpy as np
from src.mgnn.trajectories import TrajectoriesDataset
from os.path import join

torch.manual_seed(0)
np.random.seed(0)

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='eth',
                        help='eth,hotel,univ,zara1,zara2')
    parser.add_argument('--obs_seq_len', type=int, default=8)
    parser.add_argument('--pred_seq_len', type=int, default=12)
    parser.add_argument('--invalid_value', type=float, default=-999.)
    return parser.parse_args()

def main():
    args = arg_parse()
    dataset_folderpath = join(pathhack.pkg_path, 'datasets', args.dataset)
    for mode in ['train', 'val', 'test']:
        dset = TrajectoriesDataset(
            join(dataset_folderpath, 'raw'),
            obs_seq_len=args.obs_seq_len,
            pred_seq_len=args.pred_seq_len,
            skip=1,
            invalid_value=args.invalid_value,
            mode=mode,
        )
        result_filename = args.dataset+'_dset_'+mode+'_trajectories.pt'
        torch.save(dset, join(dataset_folderpath, result_filename))
        print(join(dataset_folderpath, result_filename)+' is created.')
    return

main()