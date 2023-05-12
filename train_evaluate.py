import datetime
import argparse
import numpy as np
import dgl
import torch
from models.model import HGNN
from models.utils import GraphDataset_Classification,GraphDataLoader_Classification,\
                  AUC,RMSE,\
                  GraphDataset_Regression,GraphDataLoader_Regression
from torch.optim import Adam
from data.split_data import get_classification_dataset,get_regression_dataset
    
def main(args):
    max_score_list=[]
    max_aupr_list=[]
    task_type=None
    if args.dataset in ['Tox21', 'ClinTox',
                      'SIDER', 'BBBP', 'BACE']:
        task_type='classification'
    else:
        task_type='regression'
    for seed in range(args.seed,args.seed+args.folds):
        print('folds:',seed)


        if task_type=='classification':
            metric=AUC
            train_gs,train_ls,train_tw,val_gs,val_ls,test_gs,test_ls=get_classification_dataset(args.dataset,args.n_jobs,seed,args.split_ratio)
            print(len(train_ls),len(val_ls),len(test_ls),train_tw)
            train_ds = GraphDataset_Classification(train_gs, train_ls)
            train_dl = GraphDataLoader_Classification(train_ds, num_workers=0, batch_size=args.batch_size,
                                       shuffle=args.shuffle)
            task_pos_weights=train_tw
            criterion_atom = torch.nn.BCEWithLogitsLoss(pos_weight=task_pos_weights.to(args.device))
            criterion_fg = torch.nn.BCEWithLogitsLoss(pos_weight=task_pos_weights.to(args.device))
        else:
            metric=RMSE
            train_gs, train_ls,val_gs, val_ls,test_gs,test_ls=get_regression_dataset(args.dataset,args.n_jobs,seed,args.split_ratio)
            print(len(train_ls),len(val_ls),len(test_ls))
            train_ds = GraphDataset_Regression(train_gs, train_ls)
            train_dl = GraphDataLoader_Regression(train_ds, num_workers=0, batch_size=args.batch_size,
                                       shuffle=args.shuffle)
            criterion_atom =torch.nn.MSELoss(reduction='none')
            criterion_fg =torch.nn.MSELoss(reduction='none')
            
        dist_loss=torch.nn.MSELoss(reduction='none')
        
        val_gs = dgl.batch(val_gs).to(args.device)
        val_labels=val_ls.to(args.device)
        
        test_gs=dgl.batch(test_gs).to(args.device)
        test_labels=test_ls.to(args.device)
        
        model = HGNN(val_labels.shape[1],
                      args,
                      criterion_atom,
                      criterion_fg,
                      ).to(args.device)
        print(model)
        opt = Adam(model.parameters(),lr=args.learning_rate)
        print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in model.parameters())))
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,T_0=50,eta_min=1e-4,verbose=True)
    
        
        best_val_score=0 if task_type=='classification' else 999
        best_val_aupr=0 if task_type=='classification' else 999
        best_epoch=0
        best_test_score=0
        best_test_aupr=0
        
        for epoch in range(args.epoch):
            model.train()
            traYAll = []
            traPredictAll = []
            for i, (gs, labels) in enumerate(train_dl):
                traYAll += labels.detach().cpu().numpy().tolist()
                gs = gs.to(args.device)
                labels = labels.to(args.device).float()
                af=gs.nodes['atom'].data['feat']
                bf = gs.edges[('atom', 'interacts', 'atom')].data['feat']
                fnf = gs.nodes['func_group'].data['feat']
                fef=gs.edges[('func_group', 'interacts', 'func_group')].data['feat']
                molf=gs.nodes['molecule'].data['feat']
                atom_pred,fg_pred= model(gs, af, bf,fnf,fef,molf,labels)
                ##############################################
                if task_type=='classification':
                    logits=(torch.sigmoid(atom_pred)+torch.sigmoid(fg_pred))/2
                    dist_atom_fg_loss=dist_loss(torch.sigmoid(atom_pred),torch.sigmoid(fg_pred)).mean()
                else:
                    logits=(atom_pred+fg_pred)/2
                    dist_atom_fg_loss=dist_loss(atom_pred,fg_pred).mean()
                loss_atom=criterion_atom(atom_pred,labels).mean()
                loss_motif=criterion_fg(fg_pred,labels).mean()
                loss=loss_motif+loss_atom+args.dist*dist_atom_fg_loss
                ##################################################
                opt.zero_grad()
                loss.backward()
                opt.step()
                traPredictAll += logits.detach().cpu().numpy().tolist()  
            train_score,train_AUPRC=metric(traYAll,traPredictAll)
            model.eval()
            with torch.no_grad():
                    val_af = val_gs.nodes['atom'].data['feat']
                    val_bf = val_gs.edges[('atom', 'interacts', 'atom')].data['feat']
                    val_fnf = val_gs.nodes['func_group'].data['feat']
                    val_fef=val_gs.edges[('func_group', 'interacts', 'func_group')].data['feat']
                    val_molf=val_gs.nodes['molecule'].data['feat']
                    val_logits_atom,val_logits_motif= model(val_gs, val_af, val_bf, val_fnf,val_fef,val_molf,val_labels)
            
                    test_af = test_gs.nodes['atom'].data['feat']
                    test_bf = test_gs.edges[('atom', 'interacts', 'atom')].data['feat']
                    test_fnf = test_gs.nodes['func_group'].data['feat']
                    test_fef=test_gs.edges[('func_group', 'interacts', 'func_group')].data['feat']
                    test_molf=test_gs.nodes['molecule'].data['feat']
                    test_logits_atom,test_logits_motif= model(test_gs, test_af, test_bf, test_fnf,test_fef,test_molf,test_labels)
                    ###################################################
                    if task_type=='classification':
                        val_logits=(torch.sigmoid(val_logits_atom)+torch.sigmoid(val_logits_motif))/2
                        test_logits=(torch.sigmoid(test_logits_atom)+torch.sigmoid(test_logits_motif))/2
                    else:
                        val_logits=(val_logits_atom+val_logits_motif)/2
                        test_logits=(test_logits_atom+test_logits_motif)/2
                    val_score,val_AUPRC= metric(val_labels.detach().cpu().numpy().tolist(), val_logits.detach().cpu().numpy().tolist())
                    
                    test_score,test_AUPRC=metric(test_labels.detach().cpu().numpy().tolist(), test_logits.detach().cpu().numpy().tolist())
                    ###################################################
                    if task_type=='classification':
                        if best_val_score<val_score:
                            best_val_score=val_score
                            best_test_score=test_score
                            best_epoch=epoch
                        if best_val_aupr<val_AUPRC:
                            best_val_aupr=val_AUPRC
                            best_test_aupr=test_AUPRC
                            best_epoch=epoch
                        print('#####################')
                        print("-------------------Epoch {}-------------------".format(epoch))
                        print("Train AUROC: {}".format(train_score)," Train AUPRC: {}".format(train_AUPRC))
                        print("Val AUROC: {}".format(val_score)," Val AUPRC: {}".format(val_AUPRC))
                        print("Test AUROC: {}".format(test_score)," Test AUPRC: {}".format(test_AUPRC))
                    elif task_type=='regression':
                        if best_val_score>val_score:
                            best_val_score=val_score
                            best_test_score=test_score
                            best_epoch=epoch
                        print('#####################')
                        print("-------------------Epoch {}-------------------".format(epoch))
                        print("Train RMSE: {}".format(train_score))
                        print("Val RMSE: {}".format(val_score))
                        print('Test RMSE: {}'.format(test_score))
       
        max_score_list.append(best_test_score)
        max_aupr_list.append(best_test_aupr)
        print('best model in epoch ',best_epoch)
        print('best val score is ',best_val_score)
        print('test score in this epoch is',best_test_score)
        if task_type=='classification':
            print('best val aupr is ',best_val_aupr)
            print('corresponding best test aupr is ',best_test_aupr)
    print("AUROC:\n")
    print(max_score_list)
    print(np.mean(max_score_list),'+-',np.std(max_score_list))
    print("AUPRC:\n")
    print(np.mean(max_aupr_list),'+-',np.std(max_aupr_list))
    try:
        f=open('./results/'+args.dataset+'/result_'+datetime.datetime.now().strftime('%m%d_%H%M')+'.txt','a',encoding='utf-8');  
        f.write('\n'.join([key+': '+str(value) for key, value in vars(args).items()])+'\n')
        if task_type=="classification":
            f.write("AUROC:")
        f.write(str(np.mean(max_score_list))+'+-'+str(np.std(max_score_list))+'\n')
        for i in max_score_list:
            f.write(str(i)+" ")
        if task_type=="classification":
            f.write("\nAUPRC:")
            f.write(str(np.mean(max_aupr_list))+'+-'+str(np.std(max_aupr_list))+'\n')
        f.close()
    except:
        f=open(args.dataset+'result_'+datetime.datetime.now().strftime('%m%d_%H%M')+'.txt','a',encoding='utf-8');  
        f.write('\n'.join([key+': '+str(value) for key, value in vars(args).items()])+'\n')
        if task_type=="classification":
            f.write("AUROC:")
        f.write(str(np.mean(max_score_list))+'+-'+str(np.std(max_score_list))+'\n')
        for i in max_score_list:
            f.write(str(i)+" ")
        if task_type=="classification":
            f.write("AUPRC:")
            f.write(str(np.mean(max_aupr_list))+'+-'+str(np.std(max_aupr_list))+'\n')
        f.close()
    
    



if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=str,choices=['Tox21', 'ClinTox',
                      'SIDER', 'BBBP', 'BACE','ESOL', 'FreeSolv', 'Lipophilicity'],
                   default='BBBP', help='dataset name')
    p.add_argument('--seed', type=int, default=0, help='seed used to shuffle dataset')
    p.add_argument('--atom_in_dim', type=int, default=37, help='atom feature init dim')
    p.add_argument('--bond_in_dim', type=int, default=13, help='bond feature init dim')
    p.add_argument('--ss_node_in_dim', type=int, default=50, help='func group node feature init dim')
    p.add_argument('--ss_edge_in_dim', type=int, default=37, help='func group edge feature init dim')
    p.add_argument('--mol_in_dim', type=int, default=167, help='molecule fingerprint init dim')
    p.add_argument('--learning_rate', type=float, default=5e-3, help='Adam learning rate')
    p.add_argument('--epoch', type=int, default=50, help='train epochs')
    p.add_argument('--batch_size', type=int, default=200, help='batch size for train dataset')
    p.add_argument('--num_neurons', type=list, default=[512],help='num_neurons in MLP')
    p.add_argument('--input_norm', type=str, default='layer', help='input norm')
    p.add_argument('--drop_rate', type=float, default=0.2, help='dropout rate in MLP')
    p.add_argument('--hid_dim', type=int, default=96, help='node, edge, fg hidden dims in Net')
    p.add_argument('--device', type=str, default='cuda:0', help='fitting device')
    p.add_argument('--dist',type=float,default=0.005,help='dist loss func hyperparameter lambda')
    p.add_argument('--split_ratio',type=list,default=[0.8,0.1,0.1],help='ratio to split dataset')
    p.add_argument('--folds',type=int,default=10,help='k folds validation')
    p.add_argument('--n_jobs',type=int,default=10,help='num of threads for the handle of the dataset')
    p.add_argument('--resdual',type=bool,default=False,help='resdual choice in message passing')
    p.add_argument('--shuffle',type=bool,default=False,help='whether to shuffle the train dataset')
    p.add_argument('--attention',type=bool,default=True,help='whether to use global attention pooling')
    p.add_argument('--step',type=int,default=4,help='message passing steps')
    p.add_argument('--agg_op',type=str,choices=['max','mean','sum'],default='mean',help='aggregations in local augmentation')
    p.add_argument('--mol_FP',type=str,choices=['atom','ss','both','none'],default='ss',help='cat mol FingerPrint to SS or Atom representation'
                   )
    p.add_argument('--gating_func',type=str,choices=['Softmax','Sigmoid','Identity'],default='Sigmoid',help='Gating Activation Function'
                   )
    p.add_argument('--ScaleBlock',type=str,choices=['Share','Contextual'],default='Contextual',help='Self-Rescaling Block'
                   )
    p.add_argument('--heads',type=int,default=4,help='Multi-head num')
    args = p.parse_args()
    main(args)
