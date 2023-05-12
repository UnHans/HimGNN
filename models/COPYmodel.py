import torch
import torch.nn as nn
import dgl
import math
from .MLP import MLP
from .embedding import LinearBn
from .layers import DGL_MPNNLayer
from .readout import Readout
class ContextualRescale(nn.Module):
    def __init___(self,args,atom_MLP_inDim,Motif_MLP_inDim):
        super(ContextualRescale, self).__init__()
        self.atom_scale=MLP(atom_MLP_inDim,
                             1,
                             dropout_prob=0.0,
                             num_neurons=[],input_norm=args.input_norm)
        self.fg_scale=MLP(Motif_MLP_inDim,
                             1,
                             dropout_prob=0.0,
                             num_neurons=[],input_norm=args.input_norm)
        self.modeling= MLP(2,
                           2,
                           dropout_prob=0.0,
                           num_neurons=[64],input_norm=args.input_norm)
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
        weights=self.modeling(joint_squeeze)
        weights=torch.sigmoid(weights)
        atom_scale_value=torch.reshape(weights[:,0],(atom_representation.shape[0],1))
        motif_scale_value=torch.reshape(weights[:,1],(motif_representation.shape[0],1))  
        atom_representation=atom_representation*atom_scale_value
        motif_representation=motif_representation*motif_scale_value
        return atom_representation,motif_representation
    

class TransformerRescale(nn.Module):
    def __init__(self,args,atom_MLP_inDim,SS_MLP_inDim):

        super(TransformerRescale, self).__init__()
        
        self.linear_layers = nn.ModuleList([nn.Linear(atom_MLP_inDim, atom_MLP_inDim,bias=False) for _ in range(3)])
        self.atom_output_linear = nn.Linear(atom_MLP_inDim, atom_MLP_inDim)
        self.motif_output_linear = nn.Linear(atom_MLP_inDim, atom_MLP_inDim)
        #self.W_o=nn.Linear(args.hid_dim, args.hid_dim)
        self.W_i=nn.Linear(SS_MLP_inDim,atom_MLP_inDim)
        self.heads=args.heads
        self.d_k=atom_MLP_inDim//args.heads
        if(args.gating_func=='Sigmoid'):
            self.GatingFunc=torch.sigmoid
        elif(args.gating_func=='Softmax'):
            self.GatingFunc=torch.nn.functional.softmax
        else:
            self.GatingFunc=torch.nn.Identity()
        self.dropout_layer = nn.Dropout(p=args.drop_rate)
    def forward(self,atom_representation,motif_representation):
        representation_dim=atom_representation.shape[1]
        if atom_representation.shape[1]!=motif_representation.shape[1]:
            motif_representation=self.W_i(motif_representation)
        query = []
        key = []
        value = []
        ##################
        query.append(atom_representation.unsqueeze(1))
        query.append(motif_representation.unsqueeze(1))
        key.append(atom_representation.unsqueeze(1))
        key.append(motif_representation.unsqueeze(1))
        value.append(atom_representation.unsqueeze(1))
        value.append(motif_representation.unsqueeze(1))
        ####################
        query = torch.cat(query, dim=1)
        key = torch.cat(key, dim=1)
        value = torch.cat(value, dim=1)
        ##Attention###########
        # 1) Do all the linear projections in batch from d_model => h x d_k
        batch_size=query.size(0)
        query, key, value = [l(x).view(batch_size, -1,self.heads,self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]
        # 2) Apply attention on all the projected vectors in batch.
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))
        p_attn = self.GatingFunc(scores,dim=-1)
        x=torch.matmul(p_attn, value)
        #print(p_attn)
        # 3) Split scaled_representations to atom_representation motif_representation
        scaled_representations=x.transpose(1, 2).contiguous().view(batch_size, -1, representation_dim)
        atom_representation=scaled_representations[:,0,:]
        motif_representation=scaled_representations[:,1,:]
        atom_representation=self.atom_output_linear(atom_representation)
        motif_representation=self.motif_output_linear(motif_representation)
        
        return atom_representation,motif_representation
        
        
class HMPNN(nn.Module):
    def __init__(self,
                 out_dim: int,
                 args,
                 criterion_atom,
                 criterion_fg,
                 ):
        super(HMPNN, self).__init__()
        self.args=args
        #define encoders to generate the atom and substructure embeddings
        self.atom_encoder = nn.Sequential(
            LinearBn(args.atom_in_dim,args.hid_dim),
            nn.ReLU(inplace = True),
            nn.Dropout(p =args.drop_rate),
            LinearBn(args.hid_dim,args.hid_dim),
            nn.ReLU(inplace = True)
        )
        self.fg_encoder = nn.Sequential(
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
        self.fg_mp_layer=DGL_MPNNLayer(args.hid_dim,nn.Linear(args.ss_edge_in_dim,args.hid_dim*args.hid_dim),args.resdual)
        self.atom_mp_layer=DGL_MPNNLayer(args.hid_dim,nn.Linear(args.bond_in_dim,args.hid_dim*args.hid_dim),args.resdual)
        
        #define the update function
        self.fg_update=nn.GRUCell(args.hid_dim*2,args.hid_dim)
        self.atom_update=nn.GRUCell(args.hid_dim,args.hid_dim)
        
        #define the readout layer
        self.atom_readout=Readout(args,ntype='atom',use_attention=args.attention)
        self.fg_readout=Readout(args,ntype='func_group',use_attention=args.attention)
        
        #define the predictor
        atom_MLP_inDim=args.hid_dim*2
        SS_MLP_inDim=args.hid_dim*2
        if self.mol_FP=='atom':
            atom_MLP_inDim=atom_MLP_inDim+args.mol_in_dim
        elif self.mol_FP=='ss':
            SS_MLP_inDim=SS_MLP_inDim+args.mol_in_dim
        elif self.mol_FP=='both':
            atom_MLP_inDim=atom_MLP_inDim+args.mol_in_dim
            SS_MLP_inDim=SS_MLP_inDim+args.mol_in_dim
        if args.ScaleBlock=='Contextual':
            self.scale_block=ContextualRescale(args,atom_MLP_inDim,SS_MLP_inDim)
        elif args.ScaleBlock=='Transformer':
            self.scale_block=TransformerRescale(args, atom_MLP_inDim, SS_MLP_inDim)
            SS_MLP_inDim=atom_MLP_inDim
        
        self.output_af = MLP(atom_MLP_inDim,
                                 out_dim,
                                 dropout_prob=args.drop_rate, 
                                 num_neurons=args.num_neurons,input_norm=args.input_norm)
        self.output_ff = MLP(SS_MLP_inDim,
                             out_dim,
                             dropout_prob=args.drop_rate,
                             num_neurons=args.num_neurons,input_norm=args.input_norm)
        self.criterion_atom =criterion_atom
        self.criterion_fg =criterion_fg
        self.dist_loss=torch.nn.MSELoss(reduction='none')
    def forward(self, g, af, bf, fnf, fef,mf,labels):
        with g.local_scope():
            #generate atom and substructure embeddings
            ufnf=self.fg_encoder(fnf)
            uaf=self.atom_encoder(af)
            
            #message passing and uodate
            for i in range(self.step):
                ufnm=self.fg_mp_layer(g[('func_group', 'interacts', 'func_group')],ufnf,fef)
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
                ufnm=torch.cat([ufnm,augment_ufnm],dim=1)
                
                ufnf=self.fg_update(ufnm,ufnf)
                uaf=self.atom_update(uam,uaf)
            #readout
            fg_readout=self.fg_readout(g,ufnf)
            atom_readout=self.atom_readout(g,uaf)
            ##############################
            atom_representation=atom_readout
            ss_representation=fg_readout
           
       
            ##############################
            
            if self.mol_FP=='atom':
                atom_representation=torch.cat([atom_readout,mf],dim=1)
            elif self.mol_FP=='ss':
                ss_representation=torch.cat([fg_readout,mf],dim=1)
            elif self.mol_FP=='both':
                atom_representation=torch.cat([atom_readout,mf],dim=1)
                ss_representation=torch.cat([fg_readout,mf],dim=1)
            #############################
            #print(atom_representation[0],'########################')
            atom_representation,ss_representation=self.scale_block(atom_representation,
                                                                   ss_representation)
            #print(atom_representation[0],'$$$$$$$$$$$$$$$$$$$$$$$')
            ############################
            fg_pred=self.output_ff(ss_representation)
            atom_pred=self.output_af(atom_representation)
            return atom_pred,fg_pred
