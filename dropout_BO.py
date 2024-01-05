from algorithms.dropout.dropout import Dropout
from place_db import PlaceDB
from common import grid_setting, my_inf
from utils import bo_placer, write_final_placement
import random
import argparse
import csv
import torch
import logging


def problem(node_id_ls, X, csv_writer, csv_file):
    X_ls = X.tolist()
    final_Y = []
    if len(X_ls) > 1:
        for i in range(10):
            X = X_ls[i]
            place_record = {}
            node_id_ls = node_id_ls.copy()
            for i in range(len(node_id_ls)):
                place_record[node_id_ls[i]] = {}
                place_record[node_id_ls[i]]["loc_x"] = X[i*2]
                place_record[node_id_ls[i]]["loc_y"] = X[i*2+1]
            placed_macro, hpwl = bo_placer(node_id_ls, placedb, grid_num, grid_size, place_record, csv_writer, csv_file)
            hpwl = hpwl * -1
            final_Y.append([hpwl])
    else:
        X = X_ls[0]
        place_record = {}
        node_id_ls = node_id_ls.copy()
        for i in range(len(node_id_ls)):
            place_record[node_id_ls[i]] = {}
            place_record[node_id_ls[i]]["loc_x"] = X[i*2]
            place_record[node_id_ls[i]]["loc_y"] = X[i*2+1]
        placed_macro, hpwl = bo_placer(node_id_ls, placedb, grid_num, grid_size, place_record, csv_writer, csv_file)
        hpwl = hpwl * -1
        final_Y.append([hpwl])

    final_Y = torch.Tensor(final_Y)
    return final_Y

log = logging.getLogger(__name__)
parser = argparse.ArgumentParser(description='argparse testing')
parser.add_argument('--dataset', required=True)
parser.add_argument('--seed', required=True)
args = parser.parse_args()
dataset = args.dataset
random.seed(args.seed)

hpwl_save_dir = "/home/shiyq/Guiding_EA/result/rebuttal/dropout_BO/curve/{}_seed_{}.csv".format(dataset, args.seed)
placement_save_dir = "/home/shiyq/Guiding_EA/result/rebuttal/dropout_BO/placement/{}_seed_{}.csv".format(dataset, args.seed)
rank_dir = "ranks/{}.csv".format(dataset)

hpwl_save_file = open(hpwl_save_dir,"a+")
hpwl_writer = csv.writer(hpwl_save_file)

grid_num = grid_setting[dataset]["grid_num"]
grid_size = grid_setting[dataset]["grid_size"]
placedb = PlaceDB(dataset)
macro_num = len(placedb.node_info.keys())
node_id_ls = []
with open(rank_dir) as f:
    for row in csv.reader(f):
        node_id_ls.append(row[0])

dim = 2*macro_num
algo = Dropout(dim = dim,
               lb = torch.Tensor([0 for i in range(dim)]),
               ub = torch.Tensor([grid_num for i in range(dim)]),
               active_dim = 50)

total_evals = 0
max_evals = 1000
best_hpwl = my_inf
X, Y = torch.zeros((0, dim)), torch.zeros((0, 1))

while total_evals < max_evals:
    next_X = algo.ask()
    next_Y = problem(node_id_ls, next_X, hpwl_writer, hpwl_save_file)
    
    algo.tell(next_X, next_Y)
    total_evals += len(next_X)
    next_X, next_Y = next_X.to(X), next_Y.to(Y)
    X = torch.vstack((X, next_X))
    Y = torch.vstack((Y, next_Y))
    
    results = {
        'total_evals': total_evals,
        'y': next_Y.max().item(),
        'best_y': Y.max().item(),
    }
    print(results)
    #log.info('{}'.format(results))
    #log_metrics(results, step=total_evals)