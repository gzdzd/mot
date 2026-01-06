import os
import math
import random
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import csv
import sys
import pickle

# 引入原有模块
from model import Net, NetL
from utils.config import process_config, get_args
from data_preparation import MoleculeDataset
from motif_utils import MotifGraphTeacher  # 引入上面写的模块

# 全局配置
EM_ROUNDS = 5  # EM 算法的大轮次
DIFFUSION_ITERS = 10
TEACHER_ALPHA = 0.5
DISTILLATION_LAMBDA = 1.0  # 蒸馏损失权重
LABEL_RATIO = 0.1  # 半监督设定：仅使用 10% 的标签


def get_consistency_weight(epoch, total_epochs):
    # 可选：随时间增加蒸馏权重的 Ramp-up 函数
    return DISTILLATION_LAMBDA


def train_student_em(model, device, loader, optimizer, teacher_targets, labeled_indices, dist_lambda):
    """
    M-Step: 训练学生模型
    Loss = L_sup (on labeled) + lambda * L_distill (on unlabeled)
    Ref: Appendix B.2
    """
    model.train()
    loss_all = 0
    total_samples = 0

    # teacher_targets 是一个全量 Tensor (N, )，对应 dataset 的索引
    # 我们需要在 loader 中获取当前 batch 对应的原始 dataset 索引

    for step, (bg, labels, indices) in enumerate(tqdm(loader, desc="Student Update")):
        bg = bg.to(device)
        x = bg.ndata.pop('feat')
        edge_attr = bg.edata.pop('feat')
        bases = bg.edata.pop('bases')
        labels = labels.to(device)

        # 学生预测
        pred = model(bg, x, edge_attr, bases).squeeze()

        optimizer.zero_grad()

        # 分离 Labeled 和 Unlabeled
        # indices 是当前 batch 在全集中的索引
        batch_indices = indices.cpu().numpy()
        is_labeled = np.array([idx in labeled_indices for idx in batch_indices])

        loss = torch.tensor(0.0).to(device)

        # 1. Supervised Loss (L1 or MSE) on Labeled Data
        if is_labeled.any():
            mask_l = torch.tensor(is_labeled).to(device)
            # ZINC 原代码使用 L1 Loss，Paper Appendix B 提到 Regression 用 MSE 或 NLL
            # 为了保持量级一致，这里如果原代码偏好 L1，我们也可以用 L1，或者遵从论文用 MSE
            loss_sup = F.l1_loss(pred[mask_l], labels[mask_l])
            loss += loss_sup

        # 2. Distillation Loss on Unlabeled Data (Match Teacher)
        # Appendix B: minimize (q - mu)^2  (equivalent to Gaussian NLL)
        if (~is_labeled).any():
            mask_u = torch.tensor(~is_labeled).to(device)
            # 获取对应的 Teacher Targets (q)
            batch_targets = teacher_targets[indices].to(device)

            # Gating (Optional): Paper Eq. 35 提到使用方差或一致性 gating
            # 简单起见，这里实现基础版本，不对 unlabeled 数据进行复杂 gating，直接 MSE
            loss_dist = F.mse_loss(pred[mask_u], batch_targets[mask_u])
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
        for step, (bg, labels, _) in enumerate(loader):  # 注意这里解包多了 indices
            bg = bg.to(device)
            x = bg.ndata.pop('feat')
            edge_attr = bg.edata.pop('feat')
            bases = bg.edata.pop('bases')
            labels = labels.to(device)
            pred = model(bg, x, edge_attr, bases).squeeze()
            total_mae += F.l1_loss(pred, labels, reduction='sum').item()
            count += len(labels)
    return total_mae / count


def _load_smiles_helper(dataset_name, data_dir, num_train):
    """
    [CRITICAL] 需要用户根据实际数据补充。
    从 .csv 或 .smi 文件加载训练集的 SMILES 字符串列表。
    必须保证顺序与 data_preparation.py 中加载的 graphs 顺序一致。
    """
    # 假设存在一个 train.csv 或类似文件
    # 这里为了代码不报错，生成假的 SMILES (仅作演示，实际必须替换)
    print(
        f"WARNING: Using dummy SMILES. Please implement _load_smiles_helper in main_motif_diff.py reading from {data_dir}")
    # 真实情况代码示例:
    # import pandas as pd
    # df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    # return df['smiles'].tolist()[:num_train]

    # 占位符: 如果没有真实 SMILES，Motif 提取将失败
    # 请确保你有 zinc_train.csv 之类的文件
    return ["C"] * num_train


class IndexDataset(torch.utils.data.Dataset):
    """包装 Dataset 以返回索引，便于查找 Teacher Targets"""

    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        data, label = self.dataset[index]
        return data, label, index

    def __len__(self):
        return len(self.dataset)

    def collate(self, samples):
        # 解包 (graph, label, index)
        graphs, labels, indices = map(list, zip(*samples))
        batched_graph = self.dataset.collate(list(zip(graphs, labels)))[0]  # 复用原有 collate 处理图
        return batched_graph, torch.stack(labels), torch.tensor(indices)


def main():
    args = get_args()
    config = process_config(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running MotifDiff-EM on {device}")

    # 1. 数据加载
    raw_dataset = MoleculeDataset(name=config.dataset_name, config=config.preprocess)

    # 包装 dataset 以返回 index
    train_idx_set = IndexDataset(raw_dataset.train)
    val_idx_set = IndexDataset(raw_dataset.val)
    test_idx_set = IndexDataset(raw_dataset.test)

    train_loader = DataLoader(train_idx_set, batch_size=config.hyperparams.batch_size, shuffle=True,
                              num_workers=config.hyperparams.num_workers, collate_fn=train_idx_set.collate)
    # 为 Teacher 推断准备的 loader (不 shuffle)
    train_loader_seq = DataLoader(train_idx_set, batch_size=config.hyperparams.batch_size, shuffle=False,
                                  num_workers=config.hyperparams.num_workers, collate_fn=train_idx_set.collate)

    valid_loader = DataLoader(val_idx_set, batch_size=config.hyperparams.batch_size, shuffle=False,
                              collate_fn=val_idx_set.collate)
    test_loader = DataLoader(test_idx_set, batch_size=config.hyperparams.batch_size, shuffle=False,
                             collate_fn=test_idx_set.collate)

    # 2. 半监督划分 (Simulate Semi-Supervised)
    num_train = len(raw_dataset.train)
    all_indices = np.arange(num_train)
    np.random.shuffle(all_indices)
    num_labeled = int(num_train * LABEL_RATIO)
    labeled_indices = set(all_indices[:num_labeled])
    print(f"Semi-Supervised Split: {num_labeled} Labeled, {num_train - num_labeled} Unlabeled")

    # 3. 初始化模型 (Student)
    atom_dim, bond_dim = 28, 4
    model = Net(config.architecture, num_tasks=1,
                num_basis=raw_dataset.train.graph_lists[0].edata['bases'].shape[1],
                shared_filter=config.architecture.get('shared_filter', '') == 'shd',
                linear_filter=config.architecture.get('linear_filter', '') == 'lin',
                atom_dim=atom_dim, bond_dim=bond_dim).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.hyperparams.learning_rate,
                                  weight_decay=config.hyperparams.weight_decay)

    # 4. 初始化 Teacher (Motif Graph)
    # 必须加载 SMILES，这里需要你实现 _load_smiles_helper
    train_smiles = _load_smiles_helper(config.dataset_name, './data/molecules', num_train)
    train_labels = raw_dataset.train.graph_labels.numpy()

    teacher = MotifGraphTeacher(train_smiles, train_labels, labeled_indices, alpha=TEACHER_ALPHA)

    # 初始化 Teacher Targets (全 0 或随机，第一次 E-step 会更新)
    current_teacher_targets = torch.zeros(num_train, dtype=torch.float32)

    # ================= EM 循环 =================

    # 预热: 在 Labeled 数据上先训练几个 epoch (初始化 Student)
    print(">>> Warmup Stage (Supervised only)...")
    for epoch in range(5):
        train_student_em(model, device, train_loader, optimizer, current_teacher_targets, labeled_indices,
                         dist_lambda=0.0)

    for em_round in range(EM_ROUNDS):
        print(f"\n=== EM Round {em_round + 1}/{EM_ROUNDS} ===")

        # --- E-Step: Teacher Diffusion ---
        # 1. 获取当前 Student 对所有训练数据的预测
        model.eval()
        all_preds = []
        with torch.no_grad():
            for bg, _, _ in train_loader_seq:
                bg = bg.to(device)
                x = bg.ndata.pop('feat')
                edge_attr = bg.edata.pop('feat')
                bases = bg.edata.pop('bases')
                pred = model(bg, x, edge_attr, bases).squeeze()
                all_preds.append(pred.cpu().numpy())
        student_preds_np = np.concatenate(all_preds)

        # 2. 运行扩散 (生成新的 Soft Labels Q)
        print("Running Teacher Diffusion...")
        new_targets_np = teacher.run_diffusion(student_preds_np, iterations=DIFFUSION_ITERS)
        current_teacher_targets = torch.tensor(new_targets_np, dtype=torch.float32)

        # --- M-Step: Student Distillation ---
        # 在新的 Pseudo-Labels 上训练若干 Epoch
        epochs_per_round = config.hyperparams.epochs // EM_ROUNDS
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