import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import BRICS
from collections import defaultdict
from scipy import sparse
from tqdm import tqdm

# 常见官能团的 SMARTS 模板 (论文 Section 3.4)
FUNCTIONAL_GROUP_SMARTS = {
    'Alcohol': '[#6][OX2H]',
    'Aldehyde': '[CX3H1](=O)[#6]',
    'Ketone': '[#6][CX3](=O)[#6]',
    'CarboxylicAcid': '[CX3](=O)[OX2H1]',
    'Ester': '[CX3](=O)[OX2H0][#6]',
    'Ether': '[OD2]([#6])[#6]',
    'Amine': '[NX3;H2,H1;!$(NC=O)]',
    'Amide': '[NX3][CX3](=[OX1])[#6]',
    'Nitro': '[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]',
    'Sulfonamide': '[$([#16X4](=[OX1])(=[OX1])([#6])[NX3]),$([#16X4+2]([OX1-])([OX1-])([#6])[NX3])]',
    'Halogen': '[F,Cl,Br,I]'
}


class MotifGraphTeacher:
    def __init__(self, smiles_list, labels, labeled_indices,
                 alpha=0.1, tau_min=5, tau_max_ratio=0.5, reliability_gamma=1.0):
        """
        初始化 Teacher，构建 Motif-Induced Graph。
        Ref: Section 3.5 & Appendix B
        """
        self.smiles_list = smiles_list
        self.labels = np.array(labels)
        self.labeled_indices = set(labeled_indices)
        self.num_mols = len(smiles_list)
        self.alpha = alpha
        self.fg_patterns = {name: Chem.MolFromSmarts(s) for name, s in FUNCTIONAL_GROUP_SMARTS.items()}

        # 1. 提取所有分子的 Motif
        self.mol_motifs = self._extract_all_motifs()

        # 2. 构建词表并计算 IDF (Paper Eq. 9)
        self.motif_vocab, self.motif_idf = self._build_vocab_and_idf(tau_min, tau_max_ratio)
        self.num_motifs = len(self.motif_vocab)

        # 3. 计算 Motif Reliability (Paper Eq. 12 & Appendix B)
        # 回归任务适配: 使用 labeled 样本中该 motif 对应标签的标准差衡量不确定性
        self.motif_reliability = self._compute_reliability(reliability_gamma)

        # 4. 构建矩阵 B, W, P (Paper Eq. 13-16)
        self.P = self._build_diffusion_matrix()

    def _extract_all_motifs(self):
        """使用 RDKit 提取 Bemis-Murcko Scaffolds, BRICS片段 和 官能团"""
        print("Extracting Motifs...")
        all_motifs = []
        for smi in tqdm(self.smiles_list, desc="Motif Extraction"):
            if not smi:
                all_motifs.append(set())
                continue

            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                all_motifs.append(set())
                continue

            motifs = set()
            # Method A: Bemis-Murcko Scaffold
            try:
                scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
                if scaffold: motifs.add(scaffold)
            except:
                pass

            # Method B: BRICS fragments
            try:
                frags = BRICS.BreakBRICSBonds(mol)
                frags_smi = Chem.MolToSmiles(frags).split('.')
                for f in frags_smi:
                    # 过滤简单的离子或极小片段
                    if len(f) > 1:
                        motifs.add(f)
            except:
                pass

            # Method C: Functional Groups (SMARTS)
            for name, pattern in self.fg_patterns.items():
                if pattern and mol.HasSubstructMatch(pattern):
                    # 将官能团名称作为 Motif 加入 (或者使用匹配的子结构 SMILES)
                    motifs.add(f"FG:{name}")

            all_motifs.append(motifs)
        return all_motifs

    def _build_vocab_and_idf(self, tau_min, tau_max_ratio):
        """频率过滤与 IDF 计算"""
        counter = defaultdict(int)
        for motifs in self.mol_motifs:
            for m in motifs:
                counter[m] += 1

        vocab = {}
        idf = {}
        idx = 0
        N = self.num_mols

        for m, freq in counter.items():
            if tau_min <= freq <= tau_max_ratio * N:
                vocab[m] = idx
                # Eq. 9: IDF(m) = log(N / (df(m) + epsilon))
                idf[idx] = np.log(N / (freq + 1e-5))
                idx += 1

        print(f"Motif Vocab Size: {len(vocab)} (Original: {len(counter)})")
        return vocab, idf

    def _compute_reliability(self, gamma):
        """
        计算 Reliability (r_m).
        回归任务适配: r_m = exp(-gamma * std(labels_with_motif))
        Ref: Appendix B (unchanged structure) & Section 3.4.1
        """
        r_m = np.ones(self.num_motifs)
        motif_to_labels = defaultdict(list)

        # 仅使用有标签数据计算 Reliability
        for i in self.labeled_indices:
            motifs = self.mol_motifs[i]
            val = self.labels[i]
            for m in motifs:
                if m in self.motif_vocab:
                    m_idx = self.motif_vocab[m]
                    motif_to_labels[m_idx].append(val)

        for m_idx, vals in motif_to_labels.items():
            if len(vals) > 1:
                std = np.std(vals)
                # 使用标准差代替熵
                r_m[m_idx] = np.exp(-gamma * std)
            else:
                r_m[m_idx] = 0.5  # 样本太少降低可靠性

        return r_m

    def _build_diffusion_matrix(self):
        """构建归一化的扩散矩阵 P = D^-1 W (Eq. 16)"""
        print("Building Graph Matrices...")
        row, col, data = [], [], []

        # 构建 Incidence Matrix B (Sparse)
        # B_im = 1 * IDF(m) * r_m (Eq. 13)
        for i, motifs in enumerate(self.mol_motifs):
            for m in motifs:
                if m in self.motif_vocab:
                    m_idx = self.motif_vocab[m]
                    weight = self.motif_idf[m_idx] * self.motif_reliability[m_idx]
                    row.append(i)
                    col.append(m_idx)
                    data.append(weight)

        B = sparse.csr_matrix((data, (row, col)), shape=(self.num_mols, self.num_motifs))

        # Eq. 14: D_M = diag(B^T 1)
        D_M_vec = np.array(B.sum(axis=0)).flatten()
        D_M_inv = sparse.diags(1.0 / (D_M_vec + 1e-9))

        # Eq. 14: W = B D_M^-1 B^T
        W = B @ D_M_inv @ B.T

        # Eq. 15: D = diag(W 1)
        D_vec = np.array(W.sum(axis=1)).flatten()

        # Eq. 16: P = D^-1 W (Row stochastic)
        # 处理孤立点: D_ii = 0 -> P_ii = 1
        d_inv_data = []
        rows, cols = [], []
        for i, d in enumerate(D_vec):
            if d > 1e-9:
                d_inv_data.append(1.0 / d)
                rows.append(i)
                cols.append(i)

        D_inv = sparse.csr_matrix((d_inv_data, (rows, cols)), shape=(self.num_mols, self.num_mols))
        P = D_inv @ W

        # 补全孤立点的 P_ii = 1
        row_sums = np.array(P.sum(axis=1)).flatten()
        zero_rows = np.where(row_sums < 1e-9)[0]
        if len(zero_rows) > 0:
            I_fix = sparse.csr_matrix((np.ones(len(zero_rows)), (zero_rows, zero_rows)), shape=P.shape)
            P = P + I_fix

        return P

    def run_diffusion(self, student_preds, iterations=10):
        """
        运行 Teacher 扩散过程 (E-step).
        Ref: Eq. 19 & Appendix B.1
        """
        # 1. 构建 Anchors
        anchors = np.array(student_preds).copy()  # Unlabeled 用学生预测
        # Labeled 用 Ground Truth
        for i in self.labeled_indices:
            anchors[i] = self.labels[i]

        # 2. 迭代求解不动点
        Q = anchors.copy()
        P_sparse = self.P

        for _ in range(iterations):
            # Eq. 19: Q = (1-alpha)Y + alpha P Q
            Q = (1 - self.alpha) * anchors + self.alpha * (P_sparse @ Q)

        return Q