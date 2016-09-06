__author__ = 'jamesgin'
from matplotlib import pyplot as plt
import pandas as pd
import os
import pickle
import seaborn as sns
import numpy as np

def load_plots(dir):
    res = {}
    files = os.listdir(dir)
    for f in files:
        try:
            res[f] = pickle.load(open(dir + '/' + f, 'rb'))
        except:
            pass
    return res

def get_paths(run):
    tup_list = [(x[0][0], x[1][0], x[2], x[3], x[5], x[6]) for x in run]
    return pd.DataFrame(tup_list, columns=['loss', 'val_loss', 'top1', 'mrr', 'valid_top1', 'valid_mrr'])


def plot_tops(res):
    plt.figure()
    all_res = []
    for r in res:
        # if r.startswith('0'):
        all_paths = {}
        tops = []
        # plt.figure()
        val_losses = []

        bit_res = []
        for k in res[r]:
            all_paths[k] = get_paths(res[r][k])

            best_val = all_paths[k].sort('valid_top1').iloc[-1,:]
            # print k, best_val
            val_losses.append((k, best_val['valid_top1']))
            if k % 1 == 0:
                all_paths[k] = get_paths(res[r][k])
                all_res.append(all_paths[k])
                bit_res.append(all_paths[k])
                ser = all_paths[k]['valid_top1']

                ser.name = k / 10.0
                tops.append(ser)

                # all_paths[k][['loss', 'val_loss']].plot()
                # plt.xlabel('Epoch')
                # plt.ylabel('Loss')
                # plt.title('90% Generated Data')
                # plt.savefig('images/90gen.png')

        bit_res = pd.concat(bit_res)
        print(r, bit_res.sort('val_loss').iloc[0]['top1'])
        vl = np.array(val_losses)
        print(r, val_losses)
        plt.plot(vl[:,0]/10,vl[:,1])
        plt.xlabel('Generation probability $\lambda$')
        plt.ylabel('Mean Average Precision')
        tops = pd.concat(tops, axis=1)
        plt.figure()
        tops.plot()
        plt.xlabel('Epoch')
        plt.ylabel('Validation Loss')
        plt.savefig('images/{}.png'.format(r))
        # #
        # plt.figure()
        # vl = np.array(val_losses)
        # plt.plot(vl[:,0], vl[:,1])
            # plt.savefig('images/{}_top_by_val.png'.format(r))
    # plt.legend([r.replace('p','.') for r in res.keys()])
    # plt.savefig('images/valid_top1.png')
    pass


res = load_plots('thingie')
plot_tops(res)