import dill
import pandas as pd
import numpy as np
import argparse
from collections import defaultdict
from sklearn.metrics import jaccard_score
from torch.optim import Adam
import os
import torch
import time
from models import MEGACare
from util import llprint, multi_label_metric, ddi_rate_score, get_n_params, GraphMPNN
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import BRICS

def set_seed():
    torch.manual_seed(1203)
    np.random.seed(2048)

# setting
model_name = 'MEGACare'

# resume_path = ''


if not os.path.exists(os.path.join("saved", model_name)):
        os.makedirs(os.path.join("saved", model_name))

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--Test', action='store_true', default=False, help="test mode")
parser.add_argument('--model_name', type=str, default=model_name, help="model name")
parser.add_argument('--resume_path', type=str, default=resume_path, help='resume path')
parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
# parser.add_argument('--target_ddi', type=float, default=0.06, help='target ddi')
parser.add_argument('--kp', type=float, default=0.03, help='coefficient of P signal')
parser.add_argument('--dim', type=int, default=64, help='dimension')
parser.add_argument('--thsddi', type=float, default=0.16, help='[0.16, 0.18, 0.20, 0.22]')
parser.add_argument('--thaddi', type=float, default=0.06, help='[0.02, 0.04, 0.06, 0.08]')
parser.add_argument('--debug', action='store_true', default=False)
parser.add_argument('--device', type=str, default='0')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.device

if not os.path.exists(os.path.join("saved", args.model_name)):
        os.makedirs(os.path.join("saved",  args.model_name))

# evaluate
def eval(model, data_eval, voc_size, epoch):
    model.eval()
    
    smm_record = []
    result = []
    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
    med_cnt, visit_cnt = 0, 0

    for step, input in enumerate(data_eval):
        y_gt, y_pred, y_pred_prob, y_pred_label = [], [], [], []
        for adm_idx, adm in enumerate(input):
            target_output, _ , _ , _= model(input[:adm_idx+1])
            
            y_gt_tmp = np.zeros(voc_size[2])
            y_gt_tmp[adm[2]] = 1
            y_gt.append(y_gt_tmp)

            # prediction prod
            target_output = F.sigmoid(target_output).detach().cpu().numpy()[0]
            y_pred_prob.append(target_output)
            
            # prediction med set
            y_pred_tmp = target_output.copy()
            y_pred_tmp[y_pred_tmp>=0.85] = 1
            y_pred_tmp[y_pred_tmp<0.85] = 0
            y_pred.append(y_pred_tmp)

            # prediction label
            y_pred_label_tmp = np.where(y_pred_tmp == 1)[0]
            y_pred_label.append(sorted(y_pred_label_tmp))
            visit_cnt += 1
            med_cnt += len(y_pred_label_tmp)               
       
        smm_record.append(y_pred_label)
        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(np.array(y_gt), np.array(y_pred), np.array(y_pred_prob))
                                   
        ja.append(adm_ja)
        prauc.append(adm_prauc)
        avg_p.append(adm_avg_p)
        avg_r.append(adm_avg_r)
        avg_f1.append(adm_avg_f1)
        llprint('\rtest step: {} / {}'.format(step, len(data_eval)))

    # ddi rate
    # ddi_rate = ddi_rate_score(smm_record, path='../data_ATC4/ADDI.pkl')
    # ddi_pos_rate = ddi_rate_score(smm_record, path='../data_ATC4/SDDI.pkl')
        ddi_rate = ddi_rate_score(smm_record, path='../data_ATC3/ADDI.pkl')
    ddi_pos_rate = ddi_rate_score(smm_record, path='../data_ATC3/SDDI.pkl')

    llprint('\nDDI Rate: {:.4}, DDI_pos Rate: {:.4}, Jaccard: {:.4},  PRAUC: {:.4}, AVG_PRC: {:.4}, AVG_RECALL: {:.4}, AVG_F1: {:.4}, AVG_MED: {:.4}\n'.format(
        ddi_rate, ddi_pos_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1), med_cnt / visit_cnt
    ))

    # return ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1), med_cnt / visit_cnt
    return ddi_rate, ddi_pos_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1), med_cnt / visit_cnt


def main():

    # emb_dims = 2 ** int(emb_dims)
    # LR = 10 ** LR
    # l2_regularization = 10 ** l2_regularization
    # kw = 10 ** kw
    # beta = 10 ** beta
    # batch_size = int(2 ** batch_size)
    # alpha = 10 ** alpha

    params = [emb_dims, LR, decay, kw, beta, batch_size, alpha]
    print(params)

    # load data
    data_path = '../data_ATC4/records_final_new.pkl'
    voc_path = '../data_ATC4/voc_final_new.pkl'
    ehr_adj_path = '../data_ATC4/ehr_adj_final_new.pkl'    
    ddi_adj_path_all = '../data_ATC4/ADDI.pkl'
    ddi_adj_path_pos = '../data_ATC4/SDDI.pkl'
    molecule_path = '../data_ATC4/ATC2smiles_new.pkl'
    
    device = torch.device('cuda')
    # device = torch.device('cpu')
    ehr_adj = dill.load(open(ehr_adj_path, 'rb'))
    
    ddi_adj_all = dill.load(open(ddi_adj_path_all, 'rb'))
    ddi_adj_pos = dill.load(open(ddi_adj_path_pos, 'rb'))
    data = dill.load(open(data_path, 'rb'))
    molecule = dill.load(open(molecule_path, 'rb')) 

    voc = dill.load(open(voc_path, 'rb'))  
    diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']

    if args.debug:
        data = data[:100]
    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point:split_point + eval_len]
    data_eval = data[split_point+eval_len:]

    GraphEncoder, Substructure, relation = GraphMPNN(molecule, med_voc.idx2word, 2, device)
    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))  
    model = MEGACare(voc_size, ddi_adj_pos, ddi_adj_all, GraphEncoder, Substructure, relation, emb_dim=args.dim, device=device)
    # summary(model)
    print(model)
    # model.load_state_dict(torch.load(open(args.resume_path, 'rb')))

    if args.Test:
        model.load_state_dict(torch.load(open(args.resume_path, 'rb')))
        model.to(device=device)
        tic = time.time()
        ddi_list, ja_list, prauc_list, f1_list, med_list = [], [], [], [], []
        result = []
        for i in tqdm(range(10)):
            test_sample = np.random.choice(data_test, round(len(data_test)), replace=True)
            # test_sample = data_test[:11]
            addi_rate, sddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med= eval(model, test_sample, voc_size, 0)
            result.append([addi_rate, sddi_rate, ja, avg_f1, prauc, avg_med])
        
        result = np.array(result)
        mean = result.mean(axis=0)
        std = result.std(axis=0)

        outstring = ""
        for m, s in zip(mean, std):
            outstring += "{:.4f} $\pm$ {:.4f} & ".format(m, s)

        print (outstring)
        print ('test time: {}'.format(time.time() - tic))
        return 

    model.to(device=device)
    print('parameters', get_n_params(model))
    # exit()
    optimizer = Adam(list(model.parameters()), lr=args.lr)

    # start iterations
    history = defaultdict(list)
    best_epoch, best_ja, best_f1, best_prauc = 0, 0, 0, 0

    EPOCH = 50
    for epoch in tqdm(range(EPOCH)):
        tic = time.time()
        print ('\n-------------------------- epoch {} --------------------------'.format(epoch + 1))
        
        model.train()
        
        for step, input in enumerate(data_train):

            loss = 0

            all_patient_loss_1, all_patient_loss_2, prediction_, loss_bce, loss_multi, loss_ddi, ib_z = model(seq_input)
            loss = (1 - alpha) * loss_bce + alpha * loss_multi + delta * loss_ddi

            batch = step % batch_size
            ib_z_batch[batch, :] = ib_z
            dist = torch.sum(torch.square(ib_z_batch - ib_z), dim=1)
            dist = torch.clamp(dist, 0, np.inf)
            ib_z_kernel[batch, :] = dist
            ib_z_kernel[:, batch] = dist

            if step >= batch_size:
                sigma = torch.mean(dist.detach())

                if args.renyi:
                    kernel = torch.exp(-kw * ib_z_kernel / sigma)
                    I_XT = -torch.log(torch.sum(kernel))
                    loss_info = I_XT ** 2
                else:
                    distance_contribution = -torch.mean(torch.logsumexp(-kw * ib_z_kernel / sigma, dim=1))
                    I_XT = math.log(batch_size) + distance_contribution
                    loss_info = I_XT ** 2

                loss += beta * loss_info

            for idx, adm in enumerate(input):

                seq_input = input[:idx+1]

                loss_bce_target = np.zeros((1, voc_size[2]))
                loss_bce_target[:, adm[2]] = 1

                loss_multi_target = np.full((1, voc_size[2]), -1)
                for idx, item in enumerate(adm[2]):
                    loss_multi_target[0][idx] = item
                
                all_patient_loss_1, all_patient_loss_2, prediction_, loss_bce, loss_multi, loss_ddi, ib_z = model(seq_input)
                
                loss_bce = F.binary_cross_entropy_with_logits(result, torch.FloatTensor(loss_bce_target).to(device))
                loss_multi = F.multilabel_margin_loss(F.sigmoid(result), torch.LongTensor(loss_multi_target).to(device))

                result = F.sigmoid(result).detach().cpu().numpy()[0]
                result[result >= 0.5] = 1
                result[result < 0.5] = 0
                y_label = np.where(result == 1)[0]
                              
                current_addi_rate = ddi_rate_score([[y_label]], path='../data_ATC4/ADDI.pkl')
                current_sddi_rate = ddi_rate_score([[y_label]], path='../data_ATC4/SDDI.pkl')
                
                if current_addi_rate > args.thaddi and current_sddi_rate < args.thsddi:
                    loss = 0.5 * loss_bce + 0.05 * loss_multi + 5 * loss_addi + 1 * (0.5 - loss_sddi)
                elif current_addi_rate > args.thaddi and current_sddi_rate >= args.thsddi:
                    loss = 0.5 * loss_bce + 0.05 * loss_multi + 5 * loss_addi 
                elif current_addi_rate <= args.thaddi and current_sddi_rate < args.thsddi:
                    loss = 0.5 * loss_bce + 0.05 * loss_multi + 1 * (0.5 - loss_sddi)
                else:
                    loss = 0.5 * loss_bce + 0.05 * loss_multi 

                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

            llprint('\rtraining step: {} / {}'.format(step, len(data_train)))

        print ()
        print('loss: {}, aDDI loss: {}, sDDI loss: {}, BCE loss: {}, Multi loss: {}, Tri loss: {}'.format(loss, loss_addi, loss_sddi, loss_bce, loss_multi, loss_tri))
        
        print ('------------ VAL -------------')
        
        tic2 = time.time() 
        addi_rate, sddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med = eval(model, data_test, voc_size, epoch)
        
        print ('training time: {}, test time: {}'.format(time.time() - tic, time.time() - tic2))

        history['ja'].append(ja)
        history['addi_rate'].append(addi_rate)
        history['avg_p'].append(avg_p)
        history['avg_r'].append(avg_r)
        history['avg_f1'].append(avg_f1)
        history['prauc'].append(prauc)
        history['med'].append(avg_med)

        if epoch >= 5:
            print ('ddi: {}, Med: {}, Ja: {}, F1: {}, PRAUC: {}'.format(
                np.mean(history['ddi_rate'][-5:]),
                np.mean(history['med'][-5:]),
                np.mean(history['ja'][-5:]),
                np.mean(history['avg_f1'][-5:]),
                np.mean(history['prauc'][-5:])
                ))

        torch.save(model.state_dict(), open(os.path.join('saved', args.model_name, \
            'Epoch_{}_TARGET_{:.2}_JA_{:.4}_DDI_{:.4}.model'.format(epoch, args.thaddi, ja, addi_rate)), 'wb'))

        # eval JA
        if epoch != 0 and best_ja < ja:
            best_epoch = epoch
            best_ja = ja

        print ('best_epoch: {}'.format(best_epoch))

    dill.dump(history, open(os.path.join('saved', args.model_name, 'history_{}.pkl'.format(args.model_name)), 'wb'))

if __name__ == '__main__':
    main()

    Encode_Decode_Time_BO = BayesianOptimization(
        train, {
            # 'emb_dims': (5, 8),
            # 'LR': (-5, 0),
            # 'l2_regularization': (-8, -3),
            # 'kw': (0, 1),
            # 'beta': (-4, -1),
            # 'batch_size': (4, 8),
            # 'alpha': (-4, -1),
            # 'top_k': (0, 20),
            # 'simw': (-3, -1),
        }
    )
    Encode_Decode_Time_BO.maximize()
    print(Encode_Decode_Time_BO.max)