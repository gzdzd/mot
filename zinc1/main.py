import os
import math
import random
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import csv
import sys
import pickle

sys.path.append('..')

from model import Net, NetL
from utils.config import process_config, get_args
from utils.lr import MultiStepLRWarmUp
from data_preparation import MoleculeDataset
# [新增] 引入 Teacher 定义，请确保你已经创建了 motif_utils.py
from motif_utils import MotifDiffTeacher 

torch.set_num_threads(1)

# [新增] 包装类：为了在 M-Step 中通过 Index 找到对应的伪标签
class IndexedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        graph, label = self.dataset[idx]
        return graph, label, idx

    def collate(self, samples):
        # samples is a list of (graph, label, idx)
        graphs_labels = [(s[0], s[1]) for s in samples]
        indices = [s[2] for s in samples]
        
        # 调用原始 Dataset 的 collate 处理图和标签
        batched_graph, batched_labels = self.dataset.collate(graphs_labels)
        batched_indices = torch.LongTensor(indices)
        
        return batched_graph, batched_labels, batched_indices

# [新增] 辅助函数：从原始数据加载 SMILES 用于构建 Motif 图
def load_raw_smiles(data_dir, split, split_idx_file):
    print(f"Loading raw SMILES for {split}...")
    try:
        with open(os.path.join(data_dir, f"{split}.pickle"), "rb") as f:
            raw_data = pickle.load(f)
        
        with open(split_idx_file, "r") as f:
            data_idx = [list(map(int, idx)) for idx in csv.reader(f)]
            # 根据 split.index 筛选数据
            subset_data = [raw_data[i] for i in data_idx[0]]
            
        smiles_list = []
        for item in subset_data:
            # 尝试获取 SMILES，字段名可能因数据集而异，通常是 'smiles' 或 'SMILES'
            s = item.get('smiles') or item.get('SMILES')
            if s is None:
                # 如果没有 SMILES，MotifDiff 无法工作，抛出异常
                raise ValueError("Raw data does not contain 'smiles' field needed for MotifDiff.")
            smiles_list.append(s)
        return smiles_list
    except Exception as e:
        print(f"Error loading SMILES: {e}")
        print("Using dummy SMILES for debugging (Motif graph will be invalid!)")
        # 仅供调试用的占位符
        return ["C"] * len(data_idx[0])

# [新增] 半监督损失函数
def semi_supervised_loss(pred, labels, indices, is_labeled_mask, pseudo_labels, lambda_u):
    """
    pred: (Batch, 1) 学生模型预测
    labels: (Batch, 1) 真实标签
    indices: (Batch, ) 当前 Batch 数据在全集中的索引
    is_labeled_mask: (N, ) 全局布尔掩码，标记哪些索引是有标签的
    pseudo_labels: (N, 1) 全局 Teacher 生成的伪标签
    """
    # 获取当前 Batch 的掩码
    batch_is_labeled = is_labeled_mask[indices]
    
    # 1. Supervised Loss (仅在有标签数据上计算)
    if batch_is_labeled.sum() > 0:
        loss_sup = F.l1_loss(pred[batch_is_labeled], labels[batch_is_labeled])
    else:
        loss_sup = torch.tensor(0.0).to(pred.device)

    # 2. Distillation Loss (仅在无标签数据上计算，拟合 Teacher 的 Pseudo Labels)
    # 根据论文附录 B，回归任务使用 MSE [cite: 981]
    if (~batch_is_labeled).sum() > 0:
        # 从全局伪标签中取出当前 Batch 对应的部分
        target_pseudo = pseudo_labels[indices].to(pred.device)
        loss_distill = F.mse_loss(pred[~batch_is_labeled], target_pseudo[~batch_is_labeled])
    else:
        loss_distill = torch.tensor(0.0).to(pred.device)

    return loss_sup + lambda_u * loss_distill

# [修改] M-Step 训练函数
def train_m_step(model, device, loader, optimizer, is_labeled_mask, pseudo_labels, lambda_u):
    model.train()
    loss_all = 0

    for step, (bg, labels, indices) in enumerate(tqdm(loader, desc="M-Step Train")):
        bg = bg.to(device)
        x = bg.ndata.pop('feat')
        edge_attr = bg.edata.pop('feat')
        bases = bg.edata.pop('bases')
        labels = labels.to(device)
        indices = indices.to(device) # 这里的 indices 是在全集中的位置

        pred = model(bg, x, edge_attr, bases)
        optimizer.zero_grad()

        # 计算半监督损失
        loss = semi_supervised_loss(
            pred, labels, indices, is_labeled_mask, pseudo_labels, lambda_u
        )
        
        loss.backward()
        optimizer.step()
        loss_all = loss_all + loss.detach().item()
        
    return loss_all / len(loader)

# E-Step 推理函数：获取学生模型对所有数据的当前预测
def get_student_predictions(model, device, loader):
    model.eval()
    all_preds = []
    # 确保 loader 是顺序的 (shuffle=False)
    with torch.no_grad():
        for step, (bg, labels, indices) in enumerate(tqdm(loader, desc="E-Step Infer")):
            bg = bg.to(device)
            x = bg.ndata.pop('feat')
            edge_attr = bg.edata.pop('feat')
            bases = bg.edata.pop('bases')
            
            pred = model(bg, x, edge_attr, bases)
            all_preds.append(pred.cpu())
    
    return torch.cat(all_preds, dim=0) # (N, 1)

def eval(model, device, loader):
    model.eval()
    total_mae = 0

    with torch.no_grad():
        for step, (bg, labels) in enumerate(tqdm(loader, desc="Eval iteration")):
            bg = bg.to(device)
            x = bg.ndata.pop('feat')
            edge_attr = bg.edata.pop('feat')
            bases = bg.edata.pop('bases')
            labels = labels.to(device)

            pred = model(bg, x, edge_attr, bases)
            total_mae += F.l1_loss(pred, labels).detach().item()

        acc = total_mae / (step + 1)

    return acc


import time
def main():
    args = get_args()
    config = process_config(args)
    cuda_id = os.environ.get('CUDA_VISIBLE_DEVICES')
    print(torch.cuda.get_device_name(0), cuda_id)
    print(config)
    
    # --- Config for MotifDiff-EM ---
    LABEL_RATIO = 0.1  # 10% 标签率
    LAMBDA_U = 1.0     # 蒸馏损失权重
    ALPHA = 0.5        # 扩散参数
    # -------------------------------

    algo_setting = str(config.commit_id[0:7]) + '_' + str(cuda_id) \
                   + '_MotifDiffEM_' + str(LABEL_RATIO) # 更新命名以区分实验

    csv_dir = config.directory + 'stat/'
    os.makedirs(os.path.dirname(csv_dir + algo_setting + '/'), exist_ok=True)
    path_stat_total = csv_dir + algo_setting + '/' + str(config.time_stamp) + 'stat_total.csv'
    
    # Init CSV
    if not os.path.exists(path_stat_total):
        with open(path_stat_total, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['ts_fk_algo_hp', 'seed', 'test', 'valid',
                                 'best_val_epoch', 'best_train', 'min_train_loss'])

    for seed in config.hyperparams.seeds:
        config.hyperparams.seed = seed
        config.time_stamp = int(time.time())
        ts_fk_algo_hp = algo_setting + '/T' + str(config.time_stamp) + '_S' + str(config.hyperparams.seed)

        # 运行修改后的 EM 训练 Loop
        epoch_idx, train_curve, valid_curve, test_curve, trainL_curve = run_with_given_seed_em(
            config, ts_fk_algo_hp, LABEL_RATIO, LAMBDA_U, ALPHA
        )

        with open(csv_dir + ts_fk_algo_hp + '.csv', 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['epoch', 'train', 'valid', 'test', 'train_loss'])
            csv_writer.writerows(
                np.transpose(np.array([epoch_idx, train_curve, valid_curve, test_curve, trainL_curve])))

        best_val_epoch = np.argmin(np.array(valid_curve))
        best_train = min(train_curve)
        print('Finished test: {}, Validation: {}, epoch: {}, best train: {}, best loss: {}'
              .format(test_curve[best_val_epoch], valid_curve[best_val_epoch],
                      best_val_epoch, best_train, min(trainL_curve)))

        with open(path_stat_total, 'a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([ts_fk_algo_hp, config.hyperparams.seed, test_curve[best_val_epoch], valid_curve[best_val_epoch],
                                 best_val_epoch, best_train, min(trainL_curve)])


def run_with_given_seed_em(config, ts_fk_algo_hp, label_ratio, lambda_u, alpha):
    """
    主要训练逻辑，修改为 EM 风格
    """
    if config.hyperparams.get('seed') is not None:
        random.seed(config.hyperparams.seed)
        torch.manual_seed(config.hyperparams.seed)
        np.random.seed(config.hyperparams.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.hyperparams.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = MoleculeDataset(name=config.dataset_name, config=config.preprocess)
    
    # [新增] 1. 划分 Labeled / Unlabeled
    num_train = len(dataset.train)
    indices = np.arange(num_train)
    np.random.shuffle(indices) # 随机打乱以选择有标签集
    
    n_labeled = int(label_ratio * num_train)
    labeled_indices = indices[:n_labeled]
    
    # 创建全局 Mask
    is_labeled_mask = torch.zeros(num_train, dtype=torch.bool).to(device)
    is_labeled_mask[labeled_indices] = True
    
    print(f"Semi-Supervised Split: {n_labeled} Labeled, {num_train - n_labeled} Unlabeled")

    # [新增] 2. 初始化 Teacher (MotifDiff)
    # 假设数据在 ./data/molecules/
    data_dir = './data/molecules'
    # 加载 SMILES
    train_smiles = load_raw_smiles(data_dir, 'train', os.path.join(data_dir, 'train.index'))
    
    print("Initializing MotifDiff Teacher...")
    teacher = MotifDiffTeacher(train_smiles, alpha=alpha)
    
    # 准备 Teacher 的 Anchors (N, 1)
    # 有标签部分填 Ground Truth，无标签部分后续填学生预测
    all_train_labels = torch.tensor(dataset.train.graph_labels).view(-1, 1).float()
    anchors_base = torch.zeros_like(all_train_labels)
    anchors_base[is_labeled_mask.cpu()] = all_train_labels[is_labeled_mask.cpu()]
    
    # [新增] 3. 准备 Dataloaders (注意 M-Step 和 E-Step 需求不同)
    
    # 将 train dataset 包装为 IndexedDataset
    indexed_train_set = IndexedDataset(dataset.train)
    
    # M-Step Loader: Shuffle=True, 用于训练学生
    train_loader_shuffled = DataLoader(indexed_train_set, batch_size=config.hyperparams.batch_size, shuffle=True,
                                       num_workers=config.hyperparams.num_workers, collate_fn=indexed_train_set.collate)
    
    # E-Step Loader: Shuffle=False, 用于按顺序获取预测以更新 Anchors
    train_loader_sequential = DataLoader(indexed_train_set, batch_size=config.hyperparams.batch_size, shuffle=False,
                                         num_workers=config.hyperparams.num_workers, collate_fn=indexed_train_set.collate)

    valid_loader = DataLoader(dataset.val, batch_size=config.hyperparams.batch_size, shuffle=False,
                              num_workers=config.hyperparams.num_workers, collate_fn=dataset.collate)
    test_loader = DataLoader(dataset.test, batch_size=config.hyperparams.batch_size, shuffle=False,
                             num_workers=config.hyperparams.num_workers, collate_fn=dataset.collate)

    # 4. Model Init (不变)
    atom_dim = 28
    bond_dim = 4
    if config.hyperparams.get('model', 'Net') == 'NetL':
        model = NetL(config.architecture, num_tasks=1,
                num_basis=dataset.train.graph_lists[0].edata['bases'].shape[1],
                atom_dim=atom_dim,
                bond_dim=bond_dim).to(device)
    else:
        model = Net(config.architecture, num_tasks=1,
                num_basis=dataset.train.graph_lists[0].edata['bases'].shape[1],
                shared_filter=config.architecture.get('shared_filter', '') == 'shd',
                linear_filter=config.architecture.get('linear_filter', '') == 'lin',
                atom_dim=atom_dim,
                bond_dim=bond_dim).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.hyperparams.learning_rate, betas=(0.9, 0.999), eps=1e-8,
                                  weight_decay=config.hyperparams.weight_decay)
    
    warmup_epochs = config.hyperparams.warmup_epochs
    lr_plan = lambda cur_epoch: (cur_epoch + 1) / warmup_epochs if cur_epoch < warmup_epochs else \
        (0.5 * (1.0 + math.cos(math.pi * (cur_epoch - warmup_epochs) / (config.hyperparams.epochs - warmup_epochs))))
    scheduler = LambdaLR(optimizer, lr_lambda=lr_plan)

    writer = SummaryWriter(config.directory + 'board/')
    
    # 5. Training Loop (EM Style)
    epoch_idx = []
    valid_curve = []
    test_curve = []
    train_curve = []
    trainL_curve = []
    
    # 初始 Pseudo Labels 设为 Anchors (或者全0)
    pseudo_labels = anchors_base.clone().to(device)

    print("Starting EM Training...")
    
    for epoch in range(1, config.hyperparams.epochs + 1):
        lr = scheduler.optimizer.param_groups[0]['lr']
        
        # --- E-STEP: 更新 Teacher Pseudo Labels ---
        # 仅在非 Warmup 阶段或每隔几轮执行一次（为了效率，每轮执行通常效果最好）
        # 论文中是 Alternating，这里每轮都做
        
        # 1. 获取 Student 对未标记数据的预测
        student_preds_global = get_student_predictions(model, device, train_loader_sequential) # returns CPU tensor
        
        # 2. 更新 Anchors
        current_anchors = anchors_base.clone() # 拿回带有 Labeled GT 的 anchor
        # 将 Unlabeled 部分填入学生当前的预测
        current_anchors[~is_labeled_mask.cpu()] = student_preds_global[~is_labeled_mask.cpu()]
        
        # 3. 运行扩散 (Diffusion)
        # 将数据传给 Teacher (Teacher 内部处理 GPU/CPU 移动)
        # 这里的 pseudo_labels 是经过全局图平滑后的结果 (Soft Pseudo Labels)
        pseudo_labels = teacher.propagate(current_anchors, K=10).to(device)
        
        # --- M-STEP: 训练 Student ---
        train_loss = train_m_step(
            model, device, train_loader_shuffled, optimizer, 
            is_labeled_mask, pseudo_labels, lambda_u
        )
        
        scheduler.step()

        # --- Evaluation ---
        # 注意：Eval 时只看 MAE，不需要 pseudo labels
        # 这里的 train_perf 是纯 L1 Loss (MAE)，不包含 Distillation Loss
        train_perf = eval(model, device, train_loader_sequential) 
        valid_perf = eval(model, device, valid_loader)
        test_perf = eval(model, device, test_loader)

        print('Epoch:', epoch,
              'Train MAE:', f"{train_perf:.4f}",
              'Val MAE:', f"{valid_perf:.4f}",
              'Test MAE:', f"{test_perf:.4f}",
              'Train Loss (Sup+Dist):', f"{train_loss:.4f}",
              'lr:', f"{lr:.6f}")

        epoch_idx.append(epoch)
        train_curve.append(train_perf)
        valid_curve.append(valid_perf)
        test_curve.append(test_perf)
        trainL_curve.append(train_loss)

        writer.add_scalars('traP', {ts_fk_algo_hp: train_perf}, epoch)
        writer.add_scalars('valP', {ts_fk_algo_hp: valid_perf}, epoch)
        writer.add_scalars('tstP', {ts_fk_algo_hp: test_perf}, epoch)
        writer.add_scalars('traL', {ts_fk_algo_hp: train_loss}, epoch)
        writer.add_scalars('lr',   {ts_fk_algo_hp: lr}, epoch)

        # Checkpointing (不变)
        if config.get('checkpoint_dir') is not None:
            filename_header = str(config.commit_id[0:7]) + '_' \
                       + str(config.time_stamp) + '_' \
                       + str(config.dataset_name)
            # 保存逻辑...

    writer.close()
    return epoch_idx, train_curve, valid_curve, test_curve, trainL_curve


if __name__ == "__main__":
    main()