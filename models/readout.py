import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import MultiheadAttention



class attention_readout(nn.Module):
    def __init__(self,args,ntype):
        super(attention_readout,self).__init__()
        self.ntype=ntype
        self.num_heads=args.atten_heads
        self.attention=MultiheadAttention(args.hid_dim,args.atten_heads)
        #weight=nn.Linear(args.hid_dim,1)
        #self.readout=GlobalAttentionPooling(weight)
    def forward(self,g,nf):
        attention_output ,_= self.attention(nf,None,nf)
        attention_pool=attention_output.mean(1)
        g.nodes[self.ntype].data['feat'] = attention_pool
        readout=dgl.readout_nodes(g,'feat',op='sum', ntype=self.ntype)
        #node_readout=self.readout(g,nf)
        return readout
# pylint: disable=W0221
class Readout(nn.Module):
    def __init__(self,args,ntype:str,use_attention:bool):
        super(Readout,self).__init__()
        self.ntype=ntype
        self.use_attention=use_attention
        self.linear = nn.Linear(args.hid_dim, 1)
    def forward(self,g,nf):
        if self.use_attention:
            g.nodes[self.ntype].data['nw']=self.linear(nf)
            weights=dgl.softmax_nodes(g,'nw',ntype=self.ntype)
            with g.local_scope():
                g.nodes[self.ntype].data['w'] = weights
                g.nodes[self.ntype].data['feat'] = nf
                weighted_mean_rd = dgl.readout_nodes(g, 'feat','w',op='sum', ntype=self.ntype)
                max_rd = dgl.readout_nodes(g, 'feat', op='max', ntype=self.ntype)
                return torch.cat([weighted_mean_rd,max_rd],dim=1)
        else:
            with g.local_scope():
                g.nodes[self.ntype].data['feat'] = nf
                mean_rd = dgl.readout_nodes(g, 'feat',op='mean', ntype=self.ntype)
                max_rd = dgl.readout_nodes(g, 'feat', op='max', ntype=self.ntype)
                return torch.cat([mean_rd,max_rd],dim=1)
                #return mean_rd
class GlobalPool(nn.Module):
    """One-step readout in AttentiveFP

    Parameters
    ----------
    feat_size : int
        Size for the input node features, graph features and output graph
        representations.
    dropout : float
        The probability for performing dropout.
    """
    def __init__(self, feat_size, dropout):
        super(GlobalPool, self).__init__()

        self.compute_logits = nn.Sequential(
            nn.Linear(2 * feat_size, 1),
            nn.LeakyReLU()
        )
        self.project_nodes = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_size, feat_size)
        )
        self.gru = nn.GRUCell(feat_size, feat_size)

    def forward(self, g, node_feats, g_feats,ntype, get_node_weight=False):
        """Perform one-step readout

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_feats : float32 tensor of shape (V, node_feat_size)
            Input node features. V for the number of nodes.
        g_feats : float32 tensor of shape (G, graph_feat_size)
            Input graph features. G for the number of graphs.
        get_node_weight : bool
            Whether to get the weights of atoms during readout.

        Returns
        -------
        float32 tensor of shape (G, graph_feat_size)
            Updated graph features.
        float32 tensor of shape (V, 1)
            The weights of nodes in readout.
        """
        with g.local_scope():
            g.nodes[ntype].data['z'] = self.compute_logits(
                torch.cat([dgl.broadcast_nodes(g, F.relu(g_feats),ntype=ntype), node_feats], dim=1))
            g.nodes[ntype].data['a']= dgl.softmax_nodes(g, 'z',ntype=ntype)
            g.nodes[ntype].data['hv'] = self.project_nodes(node_feats)
            g_repr = dgl.sum_nodes(g, 'hv', 'a',ntype=ntype)
            context = F.elu(g_repr)

            if get_node_weight:
                return self.gru(context, g_feats), g.ndata['a']
            else:
                return self.gru(context, g_feats)

class AttentiveFPReadout(nn.Module):
    """Readout in AttentiveFP

    AttentiveFP is introduced in `Pushing the Boundaries of Molecular Representation for
    Drug Discovery with the Graph Attention Mechanism
    <https://www.ncbi.nlm.nih.gov/pubmed/31408336>`__

    This class computes graph representations out of node features.

    Parameters
    ----------
    feat_size : int
        Size for the input node features, graph features and output graph
        representations.
    num_timesteps : int
        Times of updating the graph representations with GRU. Default to 2.
    dropout : float
        The probability for performing dropout. Default to 0.
    """
    def __init__(self, feat_size, num_timesteps=2, dropout=0.):
        super(AttentiveFPReadout, self).__init__()

        self.readouts = nn.ModuleList()
        for _ in range(num_timesteps):
            self.readouts.append(GlobalPool(feat_size, dropout))

    def forward(self, g, node_feats, ntype,etype,get_node_weight=False):
        """Computes graph representations out of node features.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_feats : float32 tensor of shape (V, node_feat_size)
            Input node features. V for the number of nodes.
        get_node_weight : bool
            Whether to get the weights of nodes in readout. Default to False.

        Returns
        -------
        g_feats : float32 tensor of shape (G, graph_feat_size)
            Graph representations computed. G for the number of graphs.
        node_weights : list of float32 tensor of shape (V, 1), optional
            This is returned when ``get_node_weight`` is ``True``.
            The list has a length ``num_timesteps`` and ``node_weights[i]``
            gives the node weights in the i-th update.
        """
        #print(node_feats.shape)
        with g.local_scope():
            g.nodes[ntype].data['hv']=node_feats
            g_feats =dgl.sum_nodes(g, 'hv',ntype=ntype)
        #print(g_feats.shape)
        if get_node_weight:
            node_weights = []

        for readout in self.readouts:
            if get_node_weight:
                g_feats, node_weights_t = readout(g, node_feats, g_feats, get_node_weight)
                node_weights.append(node_weights_t)
            else:
                g_feats = readout(g, node_feats, g_feats,ntype)

        if get_node_weight:
            return g_feats, node_weights
        else:
            return g_feats
