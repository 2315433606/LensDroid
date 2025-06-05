import os
import sys
import json
import pickle
import random
import math

import torch
from tqdm import tqdm

import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
from Define import isBackground

#root是TrainData
def get_kth_fold_data(root: str,i,k):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    train_apks_path = []  # 存储训练集的所有apk路径
    train_apks_label = []  # 存储训练集apk对应索引信息
    val_apks_path = []  # 存储验证集的所有apk路径
    val_apks_label = []  # 存储验证集apk对应索引信息
    every_class_num = []  # 存储每个类别的样本总数

    years = [year for year in os.listdir(root) if os.path.isdir(os.path.join(root, year))]
    years.sort()
    for year in years:
        # if year in ['TestStudy']:
        if year in ['data2018','data2019','data2020','data2021','data2022']:
            root_year=os.path.join(root,year,"ApkPreProcessRes")
            # 遍历文件夹，一个文件夹对应一个类别
            # apk_class = ['benign', 'malware']
            apk_class = [cla for cla in os.listdir(root_year) if os.path.isdir(os.path.join(root_year, cla))]
            # 排序，保证各平台顺序一致
            apk_class.sort()
            # 生成类别名称以及对应的数字索引
            class_indices = dict((k, v) for v, k in enumerate(apk_class))
            json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
            with open('apkPreprocess/network_all/class_indices.json', 'w') as json_file:
                json_file.write(json_str)

            # 遍历每个文件夹下的文件
            for cla in apk_class:
                #cla_path：apkPreprocess/TrainData/data2018/ApkPreProcessRes/benign/a.legalsafe.in.ahelp
                cla_path = os.path.join(root_year, cla)
                # 遍历获取apk路径
                apks = [os.path.join(root_year, cla, i) for i in os.listdir(cla_path)]
                random.shuffle(apks)

                category = class_indices[cla]

                every_class_num.append(len(apks))
                # 按比例随机采样验证样本
                fold_size = len(apks) // k  # 每份的个数:数据总条数/折数（组数）
                val_start = i * fold_size

                if i != k - 1:
                    val_end = (i + 1) * fold_size
                    val_path=apks[val_start:val_end]
                else:  # 若是最后一折交叉验证
                    val_path=apks[val_start:]

                for apk_path in apks:
                    if apk_path in val_path:  # 如果该路径在采样的验证集样本中则存入验证集
                        val_apks_path.append(apk_path)
                        val_apks_label.append(category)
                    else:  # 否则存入训练集
                        train_apks_path.append(apk_path)
                        train_apks_label.append(category)  
         
    print("{} apks were found in the dataset.".format(sum(every_class_num)))
    print("{} apks for training.".format(len(train_apks_path)))
    print("{} apks for validation.".format(len(val_apks_path)))
    assert len(train_apks_path) > 0, "number of training apks must greater than 0."
    assert len(val_apks_path) > 0, "number of validation apks must greater than 0."

    return train_apks_path, train_apks_label, val_apks_path, val_apks_label


# 在一个epoch内对模型进行训练，同时跟踪和报告训练进度和模型性能
def train_one_epoch(model, optimizer, data_loader, device, epoch, lr_scheduler, time_file):
    # 将模型设置为训练模式
    model.train()
    # 定义损失函数
    loss_function = torch.nn.CrossEntropyLoss()
    # 创建两个张量来累计训练过程中的损失和预测正确的样本数，并将它们放到指定的设备上。
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    optimizer.zero_grad() # 在每次前向传播前，需要清空优化器的梯度信息

    sample_num = 0
    #使用 tqdm 显示进度：
    if isBackground==0:
        data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        # print("  Step:",step,"Time:",time.ctime())
        images, codes, graphs, labels = data
        sample_num += images.shape[0]
        # 记录每个 batch 的开始时间
        # batch_start_time = time.time()
        # print(images.shape)
        # print(codes.shape)
        # 前向传播
        pred = model(images.to(device),codes.to(device),graphs.to(device), time_file)
        # 计算预测类别：从模型的输出 pred 中获取最大概率的索引，作为预测的类别。
        pred_classes = torch.max(pred, dim=1)[1]

        #更新准确率统计：比较预测类别和真实标签，如果相同，则在累计准确数 accu_num 上增加。
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()
        # 使用损失函数计算当前批次的损失，然后执行反向传播以计算梯度。
        loss = loss_function(pred, labels.to(device))
        loss.backward()

        #更新累计损失：将当前批次的损失添加到累计损失 accu_loss 中。使用 detach() 方法从计算图中分离损失，防止损失计算参与梯度计算。
        accu_loss += loss.detach()
        
        if isBackground==0:
            data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}, lr: {:.5f}".format(
                epoch,
                accu_loss.item() / (step + 1),
                accu_num.item() / sample_num,
                optimizer.param_groups[0]["lr"]
            )

        if not torch.isfinite(loss):
            print('\nWARNING: non-finite loss, ending training\n', loss)
            continue
            # sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()
        # update lr
        lr_scheduler.step()

        # 记录每个 batch 的结束时间
        # batch_end_time = time.time()
        # batch_times.append(batch_end_time - batch_start_time)
    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch, time_file):
    loss_function = torch.nn.CrossEntropyLoss()

    # 将模型设置为评估模式
    model.eval()

    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    # add
    ylabel=[]
    ypred=[]    

    if isBackground==0:
        data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        # print("  Step:",step,"Time:",time.ctime())
        images, codes, graphs, labels = data
        sample_num += images.shape[0]
        # 记录每个 batch 的开始时间
        # batch_start_time = time.time()
        # pred = model(images.to(device),codes.to(device),graphs.to(device))
        pred = model(images.to(device), codes.to(device), graphs.to(device), time_file)
        # pred = model(images.to(device))
        # pred = model(codes.to(device))
        # pred = model(graphs.to(device))

        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        # add 将预测的类别和真实的类别添加到相应的列表中，这些列表将用于后续的性能评估
        ylabel+=pred_classes.tolist()
        ypred+=labels.tolist()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        if isBackground==0:
            data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(
                epoch,
                accu_loss.item() / (step + 1),
                accu_num.item() / sample_num
            )
        # 记录每个 batch 的结束时间
        # batch_end_time = time.time()
        # batch_times.append(batch_end_time - batch_start_time)

    #  ylabel:实际的标签 ypred:预测的标签
    ACC = accuracy_score(ylabel, ypred)
    PRE = precision_score(ylabel, ypred)
    REC = recall_score(ylabel, ypred)
    F1 = f1_score(ylabel, ypred)
    # accu_loss.item() / (step + 1)：平均损失,即累积损失除以批次总数。   
    # accu_num.item() / sample_num：准确率,累积预测正确的样本数除以总样本数
    
    return accu_loss.item() / (step + 1), accu_num.item() / sample_num ,ACC,PRE,REC,F1

# 创建一个自定义的学习率调度器
def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3,
                        end_factor=1e-6):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            current_step = (x - warmup_epochs * num_step)
            cosine_steps = (epochs - warmup_epochs) * num_step
            # warmup后lr倍率因子从1 -> end_factor
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def get_params_groups(model: torch.nn.Module, weight_decay: float = 1e-5):
    # 记录optimize要训练的权重参数
    parameter_group_vars = {"decay": {"params": [], "weight_decay": weight_decay},
                            "no_decay": {"params": [], "weight_decay": 0.}}

    # 记录对应的权重名称
    parameter_group_names = {"decay": {"params": [], "weight_decay": weight_decay},
                             "no_decay": {"params": [], "weight_decay": 0.}}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights

        if len(param.shape) == 1 or name.endswith(".bias"):
            group_name = "no_decay"
        else:
            group_name = "decay"
            # print(name)

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)

    # print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    json_str=json.dumps(parameter_group_names, indent=2)
    with open('apkPreprocess/network_all/Param groups.json', 'w') as json_file:
        json_file.write(json_str)
    return list(parameter_group_vars.values())
