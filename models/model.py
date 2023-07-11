import torch
import torch.nn as nn
import dgl
import math
from .MLP import MLP
from .embedding import LinearBn
from .layers import DGL_MPNNLayer
from .readout import Readout
class ContextualRescale(nn.Module):
    def __init__(self,args,atom_MLP_inDim,Motif_MLP_inDim):
        super(ContextualRescale,self).__init__()
        self.atom_scale=MLP(atom_MLP_inDim,
                             1,
                             dropout_prob=0.0,
                             num_neurons=[],input_norm=args.input_norm)
        self.fg_scale=MLP(Motif_MLP_inDim,
                             1,
                             dropout_prob=0.0,
                             num_neurons=[],input_norm=args.input_norm)
        self.W1=nn.Linear(2,64)
        self.W2=nn.Linear(64,2)
        if(args.gating_func=='Sigmoid'):
            self.GatingFunc=torch.sigmoid
        elif(args.gating_func=='Softmax'):
            self.GatingFunc=torch.nn.functional.softmax
        else:
            self.GatingFunc=torch.nn.Identity()
    def forward(self,atom_representation,motif_representation):
        atom_squeeze=self.atom_scale(atom_representation)
        motif_squeeze=self.fg_scale(motif_representation)
        joint_squeeze=torch.cat([atom_squeeze,motif_squeeze],dim=1)
        weights=self.W1(joint_squeeze)
        weights=self.W2(weights)
        weights=torch.sigmoid(weights)
        atom_scale_value=torch.reshape(weights[:,0],(atom_representation.shape[0],1))
        motif_scale_value=torch.reshape(weights[:,1],(motif_representation.shape[0],1))  
        atom_representation=atom_representation*atom_scale_value
        motif_representation=motif_representation*motif_scale_value
        return atom_representation,motif_representation
    
class NormRescale(nn.Module):
    def __init__(self,args,atom_MLP_inDim,Motif_MLP_inDim):
        super(ContextualRescale,self).__init__()
        self.atom_scale=MLP(atom_MLP_inDim,
                             1,
                             dropout_prob=0.0,
                             num_neurons=[],input_norm=args.input_norm)
        self.fg_scale=MLP(Motif_MLP_inDim,
                             1,
                             dropout_prob=0.0,
                             num_neurons=[],input_norm=args.input_norm)
        if(args.gating_func=='Sigmoid'):
            self.GatingFunc=torch.sigmoid
        elif(args.gating_func=='Softmax'):
            self.GatingFunc=torch.nn.functional.softmax
        else:
            self.GatingFunc=torch.nn.Identity()
    def forward(self,atom_representation,motif_representation):
        atom_scale_value=self.atom_scale(atom_representation)
        motif_scale_value=self.fg_scale(motif_representation)
        atom_representation=atom_representation*atom_scale_value
        motif_representation=motif_representation*motif_scale_value
        return atom_representation,motif_representation

class LocalAugmentation(nn.Module):
    def __init__(self,args):
        super(LocalAugmentation,self).__init__()
        self.linear_layers = nn.ModuleList([nn.Linear(args.hid_dim, args.hid_dim,bias=False) for _ in range(3)])
        self.W_o=nn.Linear(args.hid_dim, args.hid_dim)
        self.heads=args.heads
        self.d_k=args.hid_dim//args.heads
    def forward(self,fine_messages,coarse_messages,motif_features):
        batch_size=fine_messages.shape[0]
        hid_dim=fine_messages.shape[-1]
        Q=motif_features
        K=[]
        K.append(fine_messages.unsqueeze(1))
        K.append(coarse_messages.unsqueeze(1))
        K=torch.cat(K,dim=1)
        Q=Q.view(batch_size, -1, 1,hid_dim).transpose(1, 2)
        K=K.view(batch_size, -1, 1,hid_dim).transpose(1, 2)
        V=K
        Q, K, V = [l(x).view(batch_size, -1,self.heads,self.d_k).transpose(1, 2)
                                      for l, x in zip(self.linear_layers, (Q,K,V))]   
        #print(Q[0],K.transpose(-2, -1)[0])
        message_interaction=torch.matmul( Q,K.transpose(-2, -1))/self.d_k
        #print(message_interaction[0])
        att_score=torch.nn.functional.softmax(message_interaction,dim=-1)
        motif_messages=torch.matmul(att_score, V).transpose(1, 2).contiguous().view(batch_size, -1, hid_dim)
        motif_messages=self.W_o(motif_messages)
        return motif_messages.squeeze(1)
        
        
        
        
class HGNN(nn.Module):
    def __init__(self,
                 out_dim: int,
                 args,
                 criterion_atom,
                 criterion_motif,
                 ):
        super(HGNN, self).__init__()
        self.args=args
        #define encoders to generate the atom and substructure embeddings
        self.atom_encoder = nn.Sequential(
            LinearBn(args.atom_in_dim,args.hid_dim),
            nn.ReLU(inplace = True),
            nn.Dropout(p =args.drop_rate),
            LinearBn(args.hid_dim,args.hid_dim),
            nn.ReLU(inplace = True)
        )
        self.motif_encoder = nn.Sequential(
            LinearBn(args.ss_node_in_dim,args.hid_dim),
            nn.ReLU(inplace = True),
            nn.Dropout(p =args.drop_rate),
            LinearBn(args.hid_dim,args.hid_dim),
            nn.ReLU(inplace = True)
        )
        self.step=args.step 
        self.agg_op=args.agg_op
        self.mol_FP=args.mol_FP
        #define the message passing layer 
        self.motif_mp_layer=DGL_MPNNLayer(args.hid_dim,nn.Linear(args.ss_edge_in_dim,args.hid_dim*args.hid_dim),args.resdual)
        self.atom_mp_layer=DGL_MPNNLayer(args.hid_dim,nn.Linear(args.bond_in_dim,args.hid_dim*args.hid_dim),args.resdual)
        
        #define the update function
        self.motif_update=nn.GRUCell(args.hid_dim,args.hid_dim)
        self.atom_update=nn.GRUCell(args.hid_dim,args.hid_dim)
        
        #define the readout layer
        self.atom_readout=Readout(args,ntype='atom',use_attention=args.attention)
        self.motif_readout=Readout(args,ntype='func_group',use_attention=args.attention)
        self.LA=LocalAugmentation(args)
        #define the predictor
        atom_MLP_inDim=args.hid_dim*2
        Motif_MLP_inDim=args.hid_dim*2
        if self.mol_FP=='atom':
            atom_MLP_inDim=atom_MLP_inDim+args.mol_in_dim
        elif self.mol_FP=='ss':
            Motif_MLP_inDim=Motif_MLP_inDim+args.mol_in_dim
        elif self.mol_FP=='both':
            atom_MLP_inDim=atom_MLP_inDim+args.mol_in_dim
            Motif_MLP_inDim=Motif_MLP_inDim+args.mol_in_dim
        if args.ScaleBlock=='Contextual':
            self.scale_block=ContextualRescale(args,atom_MLP_inDim,Motif_MLP_inDim)
        elif args.ScaleBlock=='Norm':
            self.scale_block=NormRescale(args, atom_MLP_inDim, Motif_MLP_inDim)
        
        self.output_af = MLP(atom_MLP_inDim,
                                 out_dim,
                                 dropout_prob=args.drop_rate, 
                                 num_neurons=args.num_neurons,input_norm=args.input_norm)
        self.output_ff = MLP(Motif_MLP_inDim,
                             out_dim,
                             dropout_prob=args.drop_rate,
                             num_neurons=args.num_neurons,input_norm=args.input_norm)
        self.criterion_atom =criterion_atom
        self.criterion_motif =criterion_motif
        self.dist_loss=torch.nn.MSELoss(reduction='none')
    def forward(self, g, af, bf, fnf, fef,mf,labels):
        with g.local_scope():
            #generate atom and substructure embeddings
            ufnf=self.motif_encoder(fnf)
            uaf=self.atom_encoder(af)
            
            #message passing and uodate
            for i in range(self.step):
                ufnm=self.motif_mp_layer(g[('func_group', 'interacts', 'func_group')],ufnf,fef)
                uam=self.atom_mp_layer(g[('atom', 'interacts', 'atom')],uaf,bf)
                g.nodes['atom'].data['_uam']=uam
                if self.agg_op=='sum':
                    g.update_all(dgl.function.copy_u('_uam','uam'),dgl.function.sum('uam','agg_uam'),\
                             etype=('atom', 'a2f', 'func_group'))
                elif self.agg_op=='max':
                    g.update_all(dgl.function.copy_u('_uam','uam'),dgl.function.max('uam','agg_uam'),\
                             etype=('atom', 'a2f', 'func_group'))
                elif self.agg_op=='mean':
                    g.update_all(dgl.function.copy_u('_uam','uam'),dgl.function.mean('uam','agg_uam'),\
                             etype=('atom', 'a2f', 'func_group'))         
                augment_ufnm=g.nodes['func_group'].data['agg_uam']
                #local augmentation
                ufnm=self.LA(augment_ufnm,ufnm,ufnf)
                #ufnm=torch.cat([ufnm,augment_ufnm],dim=1)
                
                ufnf=self.motif_update(ufnm,ufnf)
                uaf=self.atom_update(uam,uaf)
            #readout
            motif_readout=self.motif_readout(g,ufnf)
            atom_readout=self.atom_readout(g,uaf)
            ##############################
            atom_representation=atom_readout
            motif_representation=motif_readout
           
       
            ##############################
            if self.mol_FP=='atom':
                atom_representation=torch.cat([atom_readout,mf],dim=1)
            elif self.mol_FP=='ss':
                motif_representation=torch.cat([motif_readout,mf],dim=1)
            elif self.mol_FP=='both':
                atom_representation=torch.cat([atom_readout,mf],dim=1)
                motif_representation=torch.cat([motif_readout,mf],dim=1)
            #############################
            atom_representation,motif_representation=self.scale_block(
                                                        atom_representation,
                                                        motif_representation
                                                        )
            ############################
            motif_pred=self.output_ff(motif_representation)
            atom_pred=self.output_af(atom_representation)
            return atom_pred,motif_pred
