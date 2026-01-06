import os
import math
import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import csv
import sys
import pickle
import argparse

# 引入项目模块
from model import Net, NetL
from utils.config import process_config, get_args
from data_preparation import MoleculeDataset
from motif_utils import MotifGraphTeacher
from mol_to_smiles import mol_graph_to_smiles  # 引入转换工具

# 全局配置 (可移至 Config 文件)
EM_ROUNDS = 5
DIFFUSION_ITERS = 10
TEACHER_ALPHA = 0.5
DISTILLATION_LAMBDA = 1.0
LABEL_RATIO = 0.1
CONSISTENCY_KAPPA = 0.2  # 一致性门控阈值 (Appendix B.2)


def train_student_em(model, device, loader, optimizer, teacher_targets, labeled_indices, dist_lambda):
    """
    M-Step: 训练学生模型
    包含一致性门控 (Consistency Gating)
    Ref: Appendix B.2 "consistency gating from two stochastic passes"
    """
    model.train()
    loss_all = 0
    total_samples = 0

    for step, (bg, labels, indices) in enumerate(tqdm(loader, desc="Student Update")):
        bg = bg.to(device)
        x = bg.ndata.pop('feat')
        edge_attr = bg.edata.pop('feat')
        bases = bg.edata.pop('bases')
        labels = labels.to(device)

        # 1. 前向传播 (Pass 1)
        pred1 = model(bg, x, edge_attr, bases).squeeze()

        optimizer.zero_grad()
        loss = torch.tensor(0.0).to(device)

        batch_indices = indices.cpu().numpy()
        is_labeled = np.array([idx in labeled_indices for idx in batch_indices])

        # --- Labeled Loss ---
        if is_labeled.any():
            mask_l = torch.tensor(is_labeled).to(device)
            # 使用 MSE 或 L1
            loss_sup = F.mse_loss(pred1[mask_l], labels[mask_l])
            loss += loss_sup

        # --- Unlabeled Distillation with Consistency Gating ---
        if (~is_labeled).any():
            mask_u = torch.tensor(~is_labeled).to(device)
            target_u = teacher_targets[indices][mask_u].to(device)
            pred_u = pred1[mask_u]

            # Pass 2 for Uncertainty Estimation (仅对 Unlabeled 部分)
            # 保持计算图以反向传播? 通常 gating 不需要梯度，只要 mask
            with torch.no_grad():
                # 重新前向传播一次，利用 Dropout 的随机性
                # 注意：DGL图结构不变，可以直接传
                pred2_all = model(bg, x, edge_attr, bases).squeeze()
                pred2_u = pred2_all[mask_u]

            # 计算一致性差异
            uncertainty = torch.abs(pred_u - pred2_u)

            # Gating: 仅选择 uncertainty <= kappa 的样本
            gating_mask = uncertainty <= CONSISTENCY_KAPPA

            if gating_mask.any():
                # Gaussian Distillation: minimize (q - mu)^2 (Appendix B.2)
                loss_dist = F.mse_loss(pred_u[gating_mask], target_u[gating_mask])
                loss += dist_lambda * loss_dist

        loss.backward()
        optimizer.step()
        loss_all += loss.detach().item() * len(labels)
        total_samples += len(labels)

    return loss_all / total_samples


def eval_model(model, device, loader):
    model.eval()
    total_mae = 0
    count = 0
    with torch.no_grad():
        for step, (bg, labels, _) in enumerate(loader):
            bg = bg.to(device)
            x = bg.ndata.pop('feat')
            edge_attr = bg.edata.pop('feat')
            bases = bg.edata.pop('bases')
            labels = labels.to(device)
            pred = model(bg, x, edge_attr, bases).squeeze()
            total_mae += F.l1_loss(pred, labels, reduction='sum').item()
            count += len(labels)
    return total_mae / count


def _load_smiles_aligned(data_dir, split_name):
    """
    加载原始 pickle 数据并转换为 SMILES，确保与 DGL Dataset 顺序一致。
    Ref: data_preparation.py 的加载逻辑
    """
    pickle_path = os.path.join(data_dir, f"{split_name}.pickle")
    index_path = os.path.join(data_dir, f"{split_name}.index")

    print(f"Loading raw data from {pickle_path} to generate SMILES...")
    with open(pickle_path, "rb") as f:
        raw_data = pickle.load(f)

    with open(index_path, "r") as f:
        data_idx = [list(map(int, idx)) for idx in csv.reader(f)]
        # 根据 index 文件过滤和排序，与 data_preparation.py 保持一致
        filtered_data = [raw_data[i] for i in data_idx[0]]

    smiles_list = []
    print("Converting Graphs to SMILES...")
    for mol_data in tqdm(filtered_data):
        smi = mol_graph_to_smiles(mol_data)
        smiles_list.append(smi)  # 如果失败是 None，MotifTeacher 会处理

    return smiles_list


class IndexDataset(torch.utils.data.Dataset):
    """包装 Dataset 以返回索引"""

    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        data, label = self.dataset[index]
        return data, label, index

    def __len__(self):
        return len(self.dataset)

    def collate(self, samples):
        graphs, labels, indices = map(list, zip(*samples))
        # 调用原始 dataset 的 collate 处理图 batching
        batched_graph, batched_labels = self.dataset.collate(list(zip(graphs, labels)))
        return batched_graph, batched_labels, torch.tensor(indices)


def main():
    args = get_args()
    config = process_config(args)

    # 强制设置 dropout 用于一致性门控
    if 'dropout' not in config.hyperparams:
        config.hyperparams['dropout'] = 0.1
    if 'dropout' not in config.architecture:
        config.architecture['dropout'] = 0.1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running MotifDiff-EM on {device}")

    # 1. 数据加载
    # 假设数据路径在 ./data/molecules (对应 data_preparation.py)
    data_dir = './data/molecules'
    raw_dataset = MoleculeDataset(name=config.dataset_name, config=config.preprocess)

    train_idx_set = IndexDataset(raw_dataset.train)
    val_idx_set = IndexDataset(raw_dataset.val)
    test_idx_set = IndexDataset(raw_dataset.test)

    train_loader = DataLoader(train_idx_set, batch_size=config.hyperparams.batch_size, shuffle=True,
                              num_workers=config.hyperparams.num_workers, collate_fn=train_idx_set.collate)
    # 顺序 Loader 用于 Teacher 推断
    train_loader_seq = DataLoader(train_idx_set, batch_size=config.hyperparams.batch_size, shuffle=False,
                                  num_workers=config.hyperparams.num_workers, collate_fn=train_idx_set.collate)

    valid_loader = DataLoader(val_idx_set, batch_size=config.hyperparams.batch_size, shuffle=False,
                              collate_fn=val_idx_set.collate)
    test_loader = DataLoader(test_idx_set, batch_size=config.hyperparams.batch_size, shuffle=False,
                             collate_fn=test_idx_set.collate)

    # 2. 半监督划分
    num_train = len(raw_dataset.train)
    all_indices = np.arange(num_train)
    np.random.shuffle(all_indices)
    num_labeled = int(num_train * LABEL_RATIO)
    labeled_indices = set(all_indices[:num_labeled])
    print(f"Semi-Supervised Split: {num_labeled} Labeled, {num_train - num_labeled} Unlabeled")

    # 3. 初始化模型
    atom_dim, bond_dim = 28, 4
    num_basis = raw_dataset.train.graph_lists[0].edata['bases'].shape[1]

    model = Net(config.architecture, num_tasks=1,
                num_basis=num_basis,
                shared_filter=config.architecture.get('shared_filter', '') == 'shd',
                linear_filter=config.architecture.get('linear_filter', '') == 'lin',
                atom_dim=atom_dim, bond_dim=bond_dim).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.hyperparams.learning_rate,
                                  weight_decay=config.hyperparams.weight_decay)

    # 4. 初始化 Teacher (Motif Graph)
    # 加载 SMILES 并构建 Motif Graph
    train_smiles = _load_smiles_aligned(data_dir, 'train')
    train_labels = raw_dataset.train.graph_labels.numpy()

    teacher = MotifGraphTeacher(train_smiles, train_labels, labeled_indices, alpha=TEACHER_ALPHA)

    # 初始化 Teacher Targets
    current_teacher_targets = torch.zeros(num_train, dtype=torch.float32)

    # ================= EM 循环 =================

    # Warmup
    print(">>> Warmup Stage (Supervised only)...")
    for epoch in range(5):
        loss = train_student_em(model, device, train_loader, optimizer, current_teacher_targets, labeled_indices,
                                dist_lambda=0.0)
        print(f"Warmup Epoch {epoch}: Loss {loss:.4f}")

    for em_round in range(EM_ROUNDS):
        print(f"\n=== EM Round {em_round + 1}/{EM_ROUNDS} ===")

        # --- E-Step: Teacher Diffusion ---
        model.eval()
        all_preds = []
        with torch.no_grad():
            for bg, _, _ in train_loader_seq:
                bg = bg.to(device)
                x = bg.ndata.pop('feat')
                edge_attr = bg.edata.pop('feat')
                bases = bg.edata.pop('bases')
                # 预测时无需 Dropout
                pred = model(bg, x, edge_attr, bases).squeeze()
                all_preds.append(pred.cpu().numpy())
        student_preds_np = np.concatenate(all_preds)

        print("Running Teacher Diffusion...")
        new_targets_np = teacher.run_diffusion(student_preds_np, iterations=DIFFUSION_ITERS)
        current_teacher_targets = torch.tensor(new_targets_np, dtype=torch.float32)

        # --- M-Step: Student Distillation ---
        epochs_per_round = max(1, config.hyperparams.epochs // EM_ROUNDS)
        for epoch in range(epochs_per_round):
            loss = train_student_em(model, device, train_loader, optimizer,
                                    current_teacher_targets, labeled_indices,
                                    dist_lambda=DISTILLATION_LAMBDA)

            val_mae = eval_model(model, device, valid_loader)
            print(f"  [Round {em_round + 1} Epoch {epoch}] Loss: {loss:.4f} | Val MAE: {val_mae:.4f}")

    # Final Test
    test_mae = eval_model(model, device, test_loader)
    print(f"\nFinal Test MAE: {test_mae:.4f}")


if __name__ == "__main__":
    main()