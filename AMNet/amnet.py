import torch
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn.inits import reset
from gin import GIN
import torch.nn as nn

EPS = 1e-8
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def to_sparse(x, mask):
    return x[mask]


def to_dense(x, mask):
    out = x.new_zeros(tuple(mask.size()) + (x.size(-1), ))
    out[mask] = x
    return out



class AMNet(torch.nn.Module):
    """
    Atom Matching Network for atom mapping an chemical reactions.

    This network identifies atom similarity based on node embeddings and performs neighborhood consensus
    to refine correspondence predictions.

    Args:
        gnn_1 (torch.nn.Module): The first Graph Neural Network (GNN) for computing node embeddings.
        gnn_2 (torch.nn.Module): The second GNN for validating neighborhood consensus.
        num_concensus_iter (int): Number of consensus iterations.

    Attributes:
        gnn_1 (torch.nn.Module): The first GNN to compute node embeddings.
        gnn_2 (torch.nn.Module): The second GNN to validate neighborhood consensus.
        num_concensus_iter (int): Number of consensus iterations.
        mlp (torch.nn.Module): A multi-layer perceptron for refining correspondence predictions.

    Methods:
        reset_parameters(self): Reset the parameters of the GNNs and MLP.
        forward(self, x_r, edge_index_r, edge_feat_r, x_p, edge_index_p, edge_feat_p, batch_size):
            Forward pass of the AMNet for similarity and correspondence prediction.
        adjust_correspondence_matrix_for_symmetry(self, M, eq_as): Update the correspondence matrix based on equivalent atom sets.
        loss(self, M, y_r, y_p, reduction='mean'): Compute the negative log likelihood loss.
        get_equivalent_atom_mask(self, eq_as): Generate a mask for equivalent atoms in the molecule.
        acc(self, M, y_r, y_p, reduction='mean'): Compute accuracy or the number of correctly mapped atoms.
        hits_at_k(self, k, M, y_r, y_p, reduction='mean'): Compute hits@k for correspondence prediction.
        __repr__(self): Get a string representation of the AMNet.
    """
    def __init__(self, gnn_1, gnn_2, num_concensus_iter, mlp_hidden_dim=512):
        super(AMNet, self).__init__()
        self.gnn_1 = gnn_1  # The first GNN to compute node embeddings.
        self.gnn_2 = gnn_2  # The second GNN to validate neighborhood consensus.
        self.num_concensus_iter = num_concensus_iter # Number of consensus iterations.
        self.mlp = Seq(
            Lin(gnn_2.out_channels, gnn_2.out_channels),
            ReLU(),
            Lin(gnn_2.out_channels, 1),
        )
        self.nonlinearsimilarity = nn.Sequential(
            nn.Linear(gnn_1.out_channels, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, 1)
        )
        

    def reset_parameters(self):
        self.gnn_1.reset_parameters()
        self.gnn_2.reset_parameters()
        self.nonlinearsimilarity.reset_parameters()

        reset(self.mlp)


    def forward(self, x_r, edge_index_r, edge_feat_r,
                x_p, edge_index_p,edge_feat_p, batch_size):
        
        h_r = self.gnn_1(x_r, edge_index_r, edge_feat_r)
        h_p = self.gnn_1(x_p, edge_index_p, edge_feat_p)

        h_r, r_mask = to_dense_batch(h_r, batch_size, fill_value=0) #binary masks (r_mask and p_mask) indicate the original lengths of the embeddings.
        h_p, p_mask = to_dense_batch(h_p, batch_size, fill_value=0)

        B, N_r, N_feat_size = h_r.size() #batch_size, num_nodes, embedding_dim
        _, N_p, _ =  h_p.size()

        R_in, R_out = self.gnn_2.in_channels, self.gnn_2.out_channels

        #M_hat = self.nonlinearsimilarity(h_r) @ self.nonlinearsimilarity(h_p).transpose(-1, -2)
        M_hat = h_r @ h_p.transpose(-1, -2) # shape batch_size, N_r, N_p
        M_mask = r_mask.view(B, N_r, 1) & p_mask.view(B, 1, N_p) #shape B, N_r, N_p
        M_0 = torch.softmax(M_hat,dim=-1)[r_mask]

        for _ in range(self.num_concensus_iter):
            M = torch.softmax(M_hat,dim=-1)
            rand_feat_r = torch.randn((B, N_r, R_in), dtype=h_r.dtype, device=h_r.device) # shape (B, N_r, R_in)
            rand_feat_p = M.transpose(-1, -2) @ rand_feat_r #shape (B, N_p, R_in)

            rand_feat_r =to_sparse(rand_feat_r,r_mask) # to reduce memory usage
            rand_feat_p =to_sparse(rand_feat_p,p_mask) 

            o_r = self.gnn_2(rand_feat_r, edge_index_r, edge_feat_r)
            o_p = self.gnn_2(rand_feat_p, edge_index_p, edge_feat_p)

            o_r= to_dense(o_r, r_mask) # to subrtaction
            o_p = to_dense(o_p, p_mask)

            D = o_r.view(B, N_r, 1, R_out) - o_p.view(B, 1, N_p, R_out) #shape (B, N_s, N_t, R_out)
            M_hat = M_hat + self.mlp(D).squeeze(-1).masked_fill(~M_mask, 0)

        M_T = torch.softmax(M_hat,dim=-1)[r_mask]
        return M_0 , M_T
    


    def symmetrywise_correspondence_matrix(self, M, eq_as,rp_mapper):
        """
        Update the predicted correspondence matrix while considering molecule symmetry to avoid penalizing indistinguishable atoms.

        Args:
            M: The initial similarity scores matrix.
            eq_as: A list of sets of equivalent atom for product molecule.
            rp_mapper: mapper function beween atoms in reactant anf product molecule atoms

        Returns:
            Updated similarity scores matrix with adjustments for molecule symmetry.
        """

        M_by_equivalent=M.clone()

        y_r = list(range(len(M)))
        for group in eq_as:
            if len(group) > 1:
                indices = list(group)
                for r_ind, p_ind in zip(y_r, rp_mapper):
                    if p_ind in indices:
                        for cnt1 in indices:
                            for cnt2 in indices:
                                        if cnt1 != cnt2:
                                                M_by_equivalent[r_ind, p_ind] += M_by_equivalent[r_ind, p_ind]


        row_sums = M_by_equivalent.sum(dim=1)
        normalized_tensor = M_by_equivalent / row_sums.view(-1, 1)
        M_by_equivalent = normalized_tensor.to(torch.float32).requires_grad_()
        return(M_by_equivalent)
        

    def loss(self, M, y_r, rp_mapper,  reduction='mean'):
        """
        Computes the negative log-likelihood loss for correspondence prediction.
        Args:
            M  : Predicted correspondence matrix (similarity scores matrix) between nodes.
            y_r  : Ground truth  reactant atom indices.
            rp_mapper  : Ground truth product to reactant atom matcher.
            reduction: Specifies the reduction method for the computed loss.
                      Options include 'none', 'mean', and 'sum'.

        Returns:
            Negative log-likelihood loss value.
        """
        index_r = range(len(y_r))
        index_r = torch.tensor(index_r, device = device)
        val = M[index_r,rp_mapper]
        nll = -torch.log(val + EPS)
        return nll if reduction == 'none' else getattr(torch, reduction)(nll)
      
           
    def acc(self, M, y_r, rp_mapper, reduction='mean'):
        """
        Calculate the accuracy or the number of correctly mapped atoms in correspondence prediction.

        Args:
            M : Predicted correspondence matrix (similarity scores matrix) between nodes.
            y_r : Ground truth  reactant node indices.
            rp_mapper : Ground truth product to reactant atom matcher.
            reduction: Specifies the reduction method for computed accuracy.
                      Options include 'mean' (default), 'sum', or 'none'.
        Returns:
            Accuracy or the number of correctly mapped atoms.
        """
        index_r = range(len(y_r))
        index_r = torch.tensor(index_r, device = device)
        pred = M[index_r].argmax(dim=-1)
        correct = (pred == rp_mapper).sum().item()
        return correct / y_r.size(0) if reduction == 'mean' else correct
    
    def hits_at_k(self,k, M, y_r,rp_mapper, reduction='mean'):
        """
        Calculate hits@k metric for correspondence prediction.
        Args:
            k: The number of top-ranked correspondences to consider.
            M  : Predicted correspondence matrix (similarity scores matrix) between nodes.
            y_r  : Ground truth  reactant node indices.
            rp_mapper : Ground truth product to reactant atom matcher.
            reduction (str, optional): Specifies the reduction method for computed hits@k metric.
                Options include 'mean' (default), 'sum', or 'none'.

        Returns:
            Hits@k metric indicating the percentage of correctly mapped atoms in the top-k predictions.
        """
        index_r = range(len(y_r))
        index_r = torch.tensor(index_r, device = device)
        pred = M[index_r].argsort(dim=-1, descending=True)[:, :k]
        correct = (pred == rp_mapper.view(-1, 1)).sum().item()
        return correct / y_r.size(0) if reduction == 'mean' else correct
    def __repr__(self):
        return ('{}(\n'
                '    gnn_1={},\n'
                '    gnn_2={},\n'
                '    num_concensus_iter={}\n)').format(self.__class__.__name__,
                                                    self.gnn_1, self.gnn_2,
                                                    self.num_concensus_iter)
    

if __name__ =='__main__':
    from torch_geometric.data import Data, Batch

    torch.manual_seed(42)
    x = torch.randn(4, 32)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    data = Data(x=x, edge_index=edge_index)
    x, e, b = data.x, data.edge_index, data.batch
    gnn_1 = GIN(data.num_node_features, 20, num_layers=2, cat=True )
    gnn_2 = GIN(20, 100, num_layers=2, cat=True)

    model = AMNet(gnn_1, gnn_2, num_concensus_iter=10)
    M_0, M_T = model(x, e, None, x, e, None, b)
    print(M_0)
    print( M_0.size() == (data.num_nodes, data.num_nodes))

    y_r = torch.tensor([0,1, 2, 3])
    y_p = torch.tensor([0,1, 2, 3])
    M = torch.tensor([[0.5,0,0.5,0], [0,1,0,0],[0.5,0,0.5,0],[0,0,0,1]])
    rp_mapper = torch.tensor([0,1, 2, 3])
    eq_as = [{0,2}, {1},{3}]
    ll2 = model.loss(M, y_r,rp_mapper)
    print(ll2)





