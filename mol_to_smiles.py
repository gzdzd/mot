#!/usr/bin/env python3
import torch
import pickle
import argparse
import os
from rdkit import Chem
from rdkit.Chem import AllChem

# 原子类型映射 (根据ZINC数据集的常见原子类型)
# 修改后的原子类型映射 (使用标准原子序数)
# 修改后的原子类型映射 (适配 ZINC GNN Benchmark 数据集编码)
# 原始编码参考: 0:C, 1:O, 2:N, 3:F, 4:C(H1), 5:S, 6:Cl, 7:O(-), 8:N(H1+), 9:Br, ...
atom_type_to_symbol = {
    0: 'C',  1: 'O',  2: 'N',  3: 'F',  4: 'C',  5: 'S',  6: 'Cl', 7: 'O',  8: 'N',  9: 'Br',
    10: 'N', 11: 'N', 12: 'N', 13: 'N', 14: 'S', 15: 'I', 16: 'P', 17: 'O', 18: 'N', 19: 'O',
    20: 'S', 21: 'P', 22: 'P', 23: 'C', 24: 'P', 25: 'S', 26: 'C', 27: 'P'
}

# 键类型映射
bond_type_to_rdkit = {
    1: Chem.BondType.SINGLE,
    2: Chem.BondType.DOUBLE,
    3: Chem.BondType.TRIPLE,
    4: Chem.BondType.AROMATIC
}

def mol_graph_to_smiles(mol_data):
    """
    将分子图数据转换为SMILES字符串
    
    参数:
    mol_data: 包含分子图信息的字典，需要有以下键:
        - num_atom: 原子数量
        - atom_type: 原子类型列表或张量 (0-28)
        - bond_type: 键类型矩阵 (num_atom × num_atom)
    
    返回:
    smiles: SMILES字符串，如果转换失败返回None
    """
    try:
        # 1. 解析输入数据
        num_atom = mol_data['num_atom']
        
        # 转换原子类型为列表
        if torch.is_tensor(mol_data['atom_type']):
            atom_types = mol_data['atom_type'].tolist()
        else:
            atom_types = mol_data['atom_type']
        
        # 转换键类型为列表
        if torch.is_tensor(mol_data['bond_type']):
            bond_types = mol_data['bond_type'].tolist()
        else:
            bond_types = mol_data['bond_type']
        
        # 2. 创建RDKit分子对象
        mol = Chem.RWMol()
        
        # 3. 添加原子
        atoms = []
        for i in range(num_atom):
            atom_type = atom_types[i]
            if atom_type in atom_type_to_symbol:
                atom_symbol = atom_type_to_symbol[atom_type]
                atom = Chem.Atom(atom_symbol)
                atom_idx = mol.AddAtom(atom)
                atoms.append(atom_idx)
            else:
                print(f"警告: 未知的原子类型 {atom_type}")
                atom = Chem.Atom('C')  # 默认使用碳
                atom_idx = mol.AddAtom(atom)
                atoms.append(atom_idx)
        
        # 4. 添加键 (只添加上三角矩阵的键，避免重复)
        added_bonds = set()
        for i in range(num_atom):
            for j in range(i+1, num_atom):
                bond_type = bond_types[i][j]
                if bond_type > 0:
                    if bond_type in bond_type_to_rdkit:
                        rdkit_bond_type = bond_type_to_rdkit[bond_type]
                        # 检查键是否已经添加
                        if (i, j) not in added_bonds and (j, i) not in added_bonds:
                            mol.AddBond(i, j, rdkit_bond_type)
                            added_bonds.add((i, j))
                    else:
                        print(f"警告: 未知的键类型 {bond_type}")
        
        # 5. 清理和生成SMILES
        mol = mol.GetMol()
        Chem.SanitizeMol(mol)
        smiles = Chem.MolToSmiles(mol)
        
        return smiles
        
    except Exception as e:
        print(f"转换失败: {e}")
        return None

def read_pickle_data(file_path):
    """
    读取pickle文件中的分子数据
    
    参数:
    file_path: pickle文件的路径
    
    返回:
    data: 包含分子数据的列表或字典
    """
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        print(f"成功读取pickle文件: {file_path}")
        return data
    except Exception as e:
        print(f"读取pickle文件失败: {e}")
        return None

# 测试函数
if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='从pickle文件中读取分子图数据并生成SMILES')
    parser.add_argument('--file', type=str, default='d:\\mot\\zinc\\data\\molecules\\val.pickle', 
                        help='要读取的pickle文件路径')
    parser.add_argument('--num_molecules', type=int, default=10, 
                        help='要处理的分子数量')
    args = parser.parse_args()
    
    # 读取pickle文件
    data = read_pickle_data(args.file)
    
    if data is not None:
        print(f"数据类型: {type(data)}")
        print(f"数据长度: {len(data)}")
        
        # 根据数据类型处理
        if isinstance(data, list):
            # 假设列表中的每个元素是一个分子图数据字典
            print(f"开始处理前 {args.num_molecules} 个分子...")
            success_count = 0
            
            for i, mol_data in enumerate(data[:args.num_molecules]):
                print(f"\n=== 处理分子 {i+1}/{args.num_molecules} ===")
                
                # 检查分子数据是否包含必要的键
                if all(key in mol_data for key in ['num_atom', 'atom_type', 'bond_type']):
                    print(f"分子 {i+1} 包含 {mol_data['num_atom']} 个原子")
                    
                    # 生成SMILES
                    smiles = mol_graph_to_smiles(mol_data)
                    
                    if smiles:
                        print(f"✅ 成功生成SMILES: {smiles}")
                        success_count += 1
                        
                        # 验证SMILES
                        mol = Chem.MolFromSmiles(smiles)
                        if mol:
                            print(f"✅ SMILES有效，分子包含 {mol.GetNumAtoms()} 个原子和 {mol.GetNumBonds()} 个键")
                        else:
                            print("❌ SMILES无效")
                    else:
                        print("❌ 无法生成SMILES")
                else:
                    print(f"❌ 分子 {i+1} 缺少必要的键")
                    print(f"   可用键: {list(mol_data.keys())}")
            
            print(f"\n=== 处理完成 ===")
            print(f"成功生成: {success_count}/{args.num_molecules} 个SMILES")
            print(f"成功率: {success_count/args.num_molecules*100:.1f}%")
            
        elif isinstance(data, dict):
            # 如果数据是字典类型，假设它是单个分子数据
            print("数据是单个分子数据字典")
            
            if all(key in data for key in ['num_atom', 'atom_type', 'bond_type']):
                smiles = mol_graph_to_smiles(data)
                
                if smiles:
                    print(f"✅ 成功生成SMILES: {smiles}")
                    
                    # 验证SMILES
                    mol = Chem.MolFromSmiles(smiles)
                    if mol:
                        print(f"✅ SMILES有效，分子包含 {mol.GetNumAtoms()} 个原子和 {mol.GetNumBonds()} 个键")
                    else:
                        print("❌ SMILES无效")
                else:
                    print("❌ 无法生成SMILES")
            else:
                print("❌ 数据缺少必要的键")
                print(f"   可用键: {list(data.keys())}")
                
        else:
            print("❌ 不支持的数据类型")
