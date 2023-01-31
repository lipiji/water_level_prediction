# coding=utf-8
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model import Model
from data import DataLoader, batchify
import pickle
import argparse, os
import random
import pandas as pd


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--dropout', type=float)
    parser.add_argument('--embedding_dim', type=int)
    parser.add_argument('--hidden_dim', type=int)
    parser.add_argument('--layers', type=int)
    parser.add_argument('--num_class', type=int)
    parser.add_argument('--ctx_len', type=int)
    parser.add_argument('--train_data', type=str)
    parser.add_argument('--dev_data', type=str)
    parser.add_argument('--test_data', type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--print_every', type=int)
    parser.add_argument('--save_every', type=int)
    parser.add_argument('--gpuid', type=int)
    parser.add_argument('--save_dir', type=str)

    return parser.parse_args()

@torch.no_grad()
def do_eval(filename, batch_size, device, model):
    with open(filename, 'rb') as f:
        _, data_list, f_max, f_min = pickle.load(f)
    idx = 0
    batch_acm = 0.
    loss_acm = 0.

    ps, ys, ds = [], [], []
    while idx < len(data_list): 
        inp, truth, dates = batchify(data_list[idx:idx+batch_size])
        ys.append(truth)
        ds.append(dates)
        truth = truth.to(device)
        inp = inp.to(device)
        res, loss = model(inp, truth)
        res = res.to('cpu')
        ps.append(res)
        batch_acm += 1
        idx += batch_size
        loss_acm += loss.item()
    return loss_acm / batch_acm, ds, ps, ys

def do_test(filename, batch_size, device, model):
    return do_eval(filename, batch_size, device, model)

def dump_predictions(filname, ds, ps, ys):
    print("dumping...%s"%(filname))
    #Hukou_H    Xingzi_H    Duchang_H   Tangyin_H   Kangshan_H
    Hukou_H1, Hukou_H2, Xingzi_H1, Xingzi_H2, Duchang_H1, Duchang_H2, \
    Tangyin_H1, Tangyin_H2, Kangshan_H1, Kangshan_H2 = [], [], [], [], [], [], [], [], [], []
    dates = []
    for ds_, ps_, ys_ in zip(ds, ps, ys):
        for i, d in enumerate(ds_):
            date = d[-1]
            preds = ps_[:, i ,:][-1]
            truths = ys_[:, i, :][-1]
            dates.append(date)
            #Hukou_H1.append(truths[0].item())
            #Hukou_H2.append(preds[0].item())
            Xingzi_H1.append(truths[0].item())
            Xingzi_H2.append(preds[0].item())
            Duchang_H1.append(truths[1].item())
            Duchang_H2.append(preds[1].item())
            Tangyin_H1.append(truths[2].item())
            Tangyin_H2.append(preds[2].item())
            Kangshan_H1.append(truths[3].item())
            Kangshan_H2.append(preds[3].item())

    data =  {   'Date': dates,
                'Xingzi_H1':Xingzi_H1,
                'Duchang_H1':Duchang_H1,
                'Tangyin_H1':Tangyin_H1,
                'Kangshan_H1':Kangshan_H1,
                'Xingzi_H2':Xingzi_H2,
                'Duchang_H2':Duchang_H2,
                'Tangyin_H2':Tangyin_H2,
                'Kangshan_H2':Kangshan_H2,
            }

    df = pd.DataFrame(data, columns = ['Date', 'Xingzi_H1', 'Duchang_H1', 'Tangyin_H1', 'Kangshan_H1',\
                                        'Xingzi_H2', 'Duchang_H2', 'Tangyin_H2', 'Kangshan_H2'])
    df.to_excel(filname, index = False, header=True)

def run(args):
    torch.manual_seed(1234)
    if args.gpuid < 0:
        device = "cpu"
    else:
        device = torch.device("cuda:" + str(args.gpuid))
    model = Model(args.model, args.num_class, args.embedding_dim, args.hidden_dim, args.layers, \
                  args.batch_size, args.dropout, device)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr) 

    train_data = DataLoader(args.train_data, args.batch_size)
    batch_acm = 0
    loss_acm = 0.
    min_mse = 99999999
    last_epoch = train_data.epoch_id
    while train_data.epoch_id <= args.epoch:
        model.train()
        for inp, truth, _ in train_data:
            batch_acm += 1
            truth = truth.to(device)
            inp = inp.to(device)

            optimizer.zero_grad()
            res, loss = model(inp, truth)
            loss_acm += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            if batch_acm%args.print_every == -1%args.print_every:
                print ('epoch %d, batch_acm %d, loss %.3f'%(train_data.epoch_id, batch_acm, loss_acm/args.print_every))
                loss_acm = 0.

            if train_data.epoch_id > 10 and last_epoch != train_data.epoch_id:
                # eval and test and save
                print("Evaluating...", args.dev_data)
                model.eval()
                mse_eval, _, _, _ = do_eval(args.dev_data, args.batch_size, device, model)
                print("eval %.4f"%(mse_eval))
                if mse_eval < min_mse:
                    min_mse = mse_eval
                    print("Testing...", args.test_data)
                    mse_test, ds, ps, ys = do_test(args.test_data, args.batch_size, device, model)
                    dump_predictions('%s/%s_L%dS%d_epoch%d_batch_%d_%.4f.xlsx'%(args.save_dir, args.model, args.layers, args.ctx_len, \
                                        train_data.epoch_id, batch_acm, mse_test), ds, ps, ys)
                    print("eval %.4f, test %.4f"%(mse_eval, mse_test))
                model.train()

                #if batch_acm%args.save_every == -1%args.save_every:
                if not os.path.exists(args.save_dir):
                    os.mkdir(args.save_dir)
                torch.save({'args':args, 'model':model.state_dict(), 'optimizer':optimizer.state_dict()}, \
                            '%s/epoch%d_batch_%d%.4f'%(args.save_dir, train_data.epoch_id, batch_acm, mse_test))
                
                last_epoch = train_data.epoch_id

if __name__ == "__main__":
    args = parse_config()
    run(args)
    exit(0)
