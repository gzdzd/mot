import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy import sparse
from tqdm import tqdm

class MotifDiffTeacher:
    def __init__(self, molecules, alpha=0.5, r_threshold=0.0):
        """
        molecules: list of RDKit molecule objects or SMILES strings corresponding to the training set.
        alpha: diffusion restart probability (0 < alpha < 1)[cite: 404].
        """
        self.molecules = [Chem.MolFromSmiles(m) if isinstance(m, str) else m for m in molecules]
        self.num_mols = len(self.molecules)
        self.alpha = alpha
        
        # 1. Extract Motifs and Build Incidence Matrix B [cite: 277]
        print("Extracting motifs and building graph...")
        self.B, self.vocab = self._build_incidence_matrix()
        
        # 2. Compute Transition Matrix P [cite: 371]
        self.P = self._compute_transition_matrix(self.B)
        
        # Move P to GPU if possible for faster matrix multiplication
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.P_torch = self._sparse_scipy_to_torch(self.P).to(self.device)

    def _build_incidence_matrix(self):
        motif_to_id = {}
        rows = []
        cols = []
        data = []
        
        # Extract Scaffolds and Functional Groups
        for idx, mol in enumerate(tqdm(self.molecules, desc="Motif Extraction")):
            if mol is None: continue
            
            # Strategy: Bemis-Murcko Scaffold [cite: 277]
            try:
                scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                scaffold_smi = Chem.MolToSmiles(scaffold)
                motifs = [scaffold_smi]
            except:
                motifs = []
                
            # (Optional) Add Functional Groups logic here if needed
            
            for motif in motifs:
                if motif not in motif_to_id:
                    motif_to_id[motif] = len(motif_to_id)
                
                m_idx = motif_to_id[motif]
                rows.append(idx)
                cols.append(m_idx)
                data.append(1.0) # Initial weight
        
        # Build Sparse Matrix B (N x M)
        num_motifs = len(motif_to_id)
        B = sparse.coo_matrix((data, (rows, cols)), shape=(self.num_mols, num_motifs)).tocsr()
        
        # TF-IDF Weighting [cite: 307]
        # df = document frequency (number of molecules containing the motif)
        df = np.array(B.sum(axis=0)).flatten()
        idf = np.log(self.num_mols / (df + 1e-5))
        
        # Apply weights
        # Note: Paper also mentions Reliability Reweighting (r_m) based on label entropy[cite: 324].
        # Since ZINC is regression and standard datasets don't usually provide reliability priors easily,
        # we stick to IDF for this implementation.
        B_weighted = B.multiply(idf) 
        
        return B_weighted, motif_to_id

    def _compute_transition_matrix(self, B):
        # W = B * D_M^-1 * B^T [cite: 352]
        # D_M = diagonal matrix of motif degrees
        motif_degrees = np.array(B.sum(axis=0)).flatten() + 1e-9
        inv_D_M = sparse.diags(1.0 / motif_degrees)
        
        W = B.dot(inv_D_M).dot(B.T)
        
        # P = D^-1 * W [cite: 371]
        # D = diagonal matrix of molecule degrees
        mol_degrees = np.array(W.sum(axis=1)).flatten()
        # Handle isolated nodes to ensure row-stochastic [cite: 372]
        mol_degrees[mol_degrees == 0] = 1.0 
        inv_D = sparse.diags(1.0 / mol_degrees)
        
        P = inv_D.dot(W)
        return P

    def _sparse_scipy_to_torch(self, sparse_mx):
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse_coo_tensor(indices, values, shape)

    def propagate(self, anchors, K=10):
        """
        Solves Q = (1-alpha)Y + alpha * P * Q via Power Iteration[cite: 403].
        anchors: tensor of shape (N, C) or (N, 1) for regression.
        """
        anchors = anchors.to(self.device)
        Q = anchors.clone()
        
        with torch.no_grad():
            for _ in range(K):
                # Sparse matrix multiplication: P * Q
                PQ = torch.sparse.mm(self.P_torch, Q)
                Q = (1 - self.alpha) * anchors + self.alpha * PQ
        
        return Q