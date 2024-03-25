import argparse
import numpy as np
from trainandtest import tr_and_te

np.random.seed(555)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='movie', help='which dataset to use')
parser.add_argument('--n_hop', type=int, default=2, help='maximum hops')
parser.add_argument('--weight', type=float, default=1e-7, help='weight of the l2 regularization term')
parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
parser.add_argument('--n_epochs', type=int, default=10, help='the number of epochs')
parser.add_argument('--n_memory', type=int, default=16, help='size of ripple set for each hop')
parser.add_argument('--use_cuda', type=bool, default=True, help='whether to use gpu')
parser.add_argument('--vecsize', type=int, default=16, help='word vector size')
parser.add_argument('--dicsize', type=int, default=1000, help='size of the dict size')
parser.add_argument('--sersize', type=int, default=16, help='user truncated series size')
parser.add_argument('--genusernum', type=int, default=4775, help='the number of normal users')
parser.add_argument('--atts', type=int, default=50, help='the number of attackers')


args = parser.parse_args()
r,p,f1score,fpr=tr_and_te(args)


