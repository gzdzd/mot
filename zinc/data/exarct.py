import pickle
# 假设你的文件路径是 data/molecules/train.pickle
with open("data/molecules/train.pickle", "rb") as f:
    data = pickle.load(f)
    print(data[0].keys()) # 查看第一个样本包含哪些字段