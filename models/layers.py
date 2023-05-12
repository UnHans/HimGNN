from functools import partial
import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import dgl.nn.pytorch as dglnn
from dgl.nn import NNConv
class Down_to_Up_aggLayer(nn.Module):
    """aggregate messages to higher-level"""
    def __init__(self,
                 args,
                 g,
                 etype):
        super(Down_to_Up_aggLayer, self).__init__()
        self.subgraph=dgl.edge_type_subgraph(g,[etype])
        self.g=g
        self.update=nn.GRUCell(args.hid_dim,args.hid_dim).to(args.device)
        self.etype=etype
    def forward(self,down_nf,down_ntype,up_ntype):
        self.g.nodes[down_ntype].data['_dnf']=down_nf
        self.g.update_all(dgl.function.copy_u('_dnf','dnf'),dgl.function.sum('dnf','agg_dnf'),\
                     etype=self.etype)
        self.g.apply_nodes(self.node_update,ntype=up_ntype)
        
        
        unf=self.g.nodes[up_ntype].data['unf']
        """
        initial_unf=unf
        with self.subgraph.local_scope():
            src_nf={down_ntype:down_nf}
            dst_nf={up_ntype:unf}
            atten_f=self.GATconv(self.subgraph,(src_nf,dst_nf))
            unf=atten_f[up_ntype].mean(1)
        update_unf=self.update(unf,initial_unf)
        return update_unf
        """
        return unf
    def node_update(self,nodes):
        agg_down_nf=nodes.data['agg_dnf']
        return {'unf':agg_down_nf}
    
    
class DGL_MPNNLayer(nn.Module):
    """message passing layer"""
    def __init__(self,
                 hid_dim,
                 edge_func,
                 resdual,
                 ):
        super(DGL_MPNNLayer, self).__init__()
        self.hidden_dim=hid_dim
        self.node_conv=NNConv(self.hidden_dim,self.hidden_dim,edge_func,'sum',resdual)
        
    def forward(self,g,nf,initial_ef):
        unm=self.node_conv(g,nf,initial_ef)
        return unm
   
       
class Self_WriteMPNNlayer(nn.Module):
    def __init__(self,
                 node_decoder: nn.Module,
                 ):
        super(Self_WriteMPNNlayer, self).__init__()
        self.W_o=node_decoder
        self.dropout_layer = nn.Dropout(p=0.1)
        self.act_func=nn.LeakyReLU(0.1)               
    def forward(self, g, nm, em,af):
        with g.local_scope():
            g.ndata['_m'] = nm
            g.edata['_m'] = em
            g.ndata['_f']=af
            g.apply_edges(func=partial(self.edge_update))
            
            # update nodes
            g.pull(g.nodes(),
                   message_func=fn.copy_e('edge_message', 'edge_message'),
                   reduce_func=fn.sum('edge_message', 'agg_em'),
                   apply_node_func=self.node_update)
            updated_nf = g.ndata['uh']
            return updated_nf

    def edge_update(self, edges):
        sender_nm = edges.src['_m']
        em = edges.data['_m']
        edge_message= torch.cat([sender_nm,em],dim=1)
        return {'edge_message': edge_message, 'uh':em}

    def node_update(self, nodes):
        agg_em = nodes.data['agg_em']
        nf=nodes.data['_f']
        nm=self.W_o(agg_em)
        update_nf=self.act_func(nm+nf)
        return {'uh': update_nf}
    

