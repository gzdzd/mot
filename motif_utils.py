import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import BRICS
from collections import defaultdict
from scipy import sparse
from tqdm import tqdm


class MotifGraphTeacher:
    def __init__(self, smiles_list, labels, labeled_indices,
                 alpha=0.1, tau_min=5, tau_max_ratio=0.5, reliability_gamma=1.0):
        """
        初始化 Teacher，构建 Motif-Induced Graph。

        Args:
            smiles_list: 所有训练样本的 SMILES 列表
            labels: 所有训练样本的标签 (Tensor or Array)
            labeled_indices: 有标签样本的索引列表
            alpha: 扩散系数 (Paper Eq. 18)
            tau_min: Motif 最小频次过滤
            tau_max_ratio: Motif 最大频次比例过滤
            reliability_gamma: Reliability 计算系数
        """
        self.smiles_list = smiles_list
        self.labels = np.array(labels)
        self.labeled_indices = set(labeled_indices)
        self.num_mols = len(smiles_list)
        self.alpha = alpha

        # 1. 提取所有分子的 Motif
        self.mol_motifs = self._extract_all_motifs()

        # 2. 构建词表并计算 IDF (Paper Eq. 9)
        self.motif_vocab, self.motif_idf = self._build_vocab_and_idf(tau_min, tau_max_ratio)
        self.num_motifs = len(self.motif_vocab)

        # 3. 计算 Motif Reliability (Paper Eq. 12 & Appendix B)
        # 回归任务适配: 使用 labeled 样本中该 motif 对应标签的标准差的倒数来衡量可靠性
        self.motif_reliability = self._compute_reliability(reliability_gamma)

        # 4. 构建矩阵 B, W, P (Paper Eq. 13-16)
        self.P = self._build_diffusion_matrix()

    def _extract_all_motifs(self):
        """使用 RDKit 提取 Bemis-Murcko Scaffolds 和 Functional Groups"""
        print("Extracting Motifs...")
        all_motifs = []
        for smi in tqdm(self.smiles_list, desc="Motif Extraction"):
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

            # Method B: BRICS fragments (作为 Functional Groups 的近似)
            # 也可以使用 RDKit 的 FunctionalGroups 库，这里简化使用 BRICS
            try:
                frags = BRICS.BreakBRICSBonds(mol)
                frags_smi = Chem.MolToSmiles(frags).split('.')
                for f in frags_smi:
                    motifs.add(f)
            except:
                pass

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
        Paper Eq. 12 原型为 exp(-gamma * Entropy)，回归用 std 代替 entropy。
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
                r_m[m_idx] = np.exp(-gamma * std)
            else:
                r_m[m_idx] = 0.5  # 默认值，如果样本太少

        return r_m

    def _build_diffusion_matrix(self):
        """构建归一化的扩散矩阵 P = D^-1 W"""
        print("Building Graph Matrices...")
        row, col, data = [], [], []

        # 构建 Incidence Matrix B (Sparse)
        # B_im = 1 * IDF(m) * r_m
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
            else:
                # 孤立点，在 P 中设对角为 1
                pass

        D_inv = sparse.csr_matrix((d_inv_data, (rows, cols)), shape=(self.num_mols, self.num_mols))
        P = D_inv @ W

        # 补全孤立点的 P_ii = 1 (由于上面 W 孤立点全0, P 也是全0, 需要加 I)
        # 简单处理：找到全0行，加1
        row_sums = np.array(P.sum(axis=1)).flatten()
        zero_rows = np.where(row_sums < 1e-9)[0]
        if len(zero_rows) > 0:
            I_fix = sparse.csr_matrix((np.ones(len(zero_rows)), (zero_rows, zero_rows)), shape=P.shape)
            P = P + I_fix

        return P

    def run_diffusion(self, student_preds, iterations=10):
        """
        运行 Teacher 扩散过程 (E-step).
        Paper Appendix B (Regression):
        Anchors: y_i if labeled, mu_theta(G_i) if unlabeled
        Diffusion: q = (1-alpha)*anchors + alpha * P * q
        """
        # 1. 构建 Anchors
        anchors = np.array(student_preds).copy()  # 初始化为学生预测 (Unlabeled 部分)
        # 覆盖 Labeled 部分为 Ground Truth
        for i in self.labeled_indices:
            anchors[i] = self.labels[i]

        # 2. 迭代求解不动点
        Q = anchors.copy()
        P_sparse = self.P

        for _ in range(iterations):
            # Eq. 19: Q = (1-alpha)Y + alpha P Q
            Q = (1 - self.alpha) * anchors + self.alpha * (P_sparse @ Q)

        return Q