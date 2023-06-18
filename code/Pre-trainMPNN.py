from models import MolecularGraphNeuralNetwork
from torch.optim import Adam
from util import llprint, multi_label_metric, ddi_rate_score, get_n_params, GraphMPNN
import torch
from util import buildMPNN
import dill

ddi_adj_path_all = 'ADDI.pkl'
ddi_adj_path_pos = 'SDDI.pkl'

ddi_adj_all = dill.load(open(ddi_adj_path_all, 'rb'))
ddi_adj_pos = dill.load(open(ddi_adj_path_pos, 'rb'))
device = torch.device('cpu')

molecule_path = 'ATC2smiles_new.pkl'
molecule = dill.load(open(molecule_path, 'rb')) 

# voc_path = 'data_ATC4/voc_final_new.pkl'
voc_path = 'data_ATC3/voc_final_new.pkl'
voc = dill.load(open(voc_path, 'rb'))
diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']


def train_MPNN(ddi_adj_pn, ddi_adj_all, emb_dim, device, train_MPNN_nums):
    
    MPNNSet_new, N_fingerprint_new, average_projection_new = buildMPNN(molecule, med_voc.idx2word, radius=1, device="cpu")
    MPNN_molecule_Set = list(zip(*MPNNSet_new))
    MPNN_net = MolecularGraphNeuralNetwork(MPNN_molecule_Set, average_projection_new, ddi_adj_all, ddi_adj_pn, N_fingerprint_new, emb_dim, layer_hidden=2, device="cpu")
    print('parameters', get_n_params(MPNN_net))
    optimizer = Adam(list(MPNN_net.parameters()), lr=1e-3)
     
    for i in range(train_MPNN_nums):
        print("epoch:", i)
        MPNN_embeds=MPNN_net()
        triple_loss = MPNN_net.triplet_loss(MPNN_embeds)
        optimizer.zero_grad()
        triple_loss.backward()
        optimizer.step()

        print(triple_loss)

    torch.save(MPNN_embeds, 'MPNN_net_embeddings_{}'.format(train_MPNN_nums))       

train_MPNN(ddi_adj_pos, ddi_adj_all, 64, device, 1000)