import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from my_dataset import MyDataSet
from model_convNeXt import convnext_tiny as create_convNeXt
from model_textCNN import create_textCNN
from model_GCN import create_GCN
from model_featherFusion import create_Fusion

from utils import create_lr_scheduler, get_params_groups, train_one_epoch, evaluate
from utils import get_kth_fold_data
import time


def train_1fold(args,i,K):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    if os.path.exists("apkPreprocess/network_all/convNeXt_weights") is False:
        os.makedirs("apkPreprocess/network_all/convNeXt_weights")

    #创建一个 TensorBoard 记录器，它可以将训练和验证过程中的各种指标和数据写入到 TensorBoard 日志文件中
    tb_writer = SummaryWriter()

    train_apks_path, train_apks_label, val_apks_path, val_apks_label = get_kth_fold_data(args.data_path,i,K)
    # print(sum(train_apks_label))
    img_size = 224
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(img_size),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(int(img_size * 1.143)),
                                   transforms.CenterCrop(img_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # 实例化训练数据集
    train_dataset = MyDataSet(apks_path=train_apks_path,
                              apks_class=train_apks_label,
                              transform=data_transform["train"])

    # 实例化验证数据集
    val_dataset = MyDataSet(apks_path=val_apks_path,
                            apks_class=val_apks_label,
                            transform=data_transform["val"])

    batch_size = args.batch_size
    # nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    nw = 16
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    model_convNeXt = create_convNeXt(num_classes=args.num_classes)
    model_textCNN = create_textCNN(num_classes=args.num_classes)
    model_GCN = create_GCN(num_classes=args.num_classes)

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)["model"]
        # 删除有关分类类别的权重
        for k in list(weights_dict.keys()):
            if "head" in k:
                del weights_dict[k]
        # print(model_convNeXt.load_state_dict(weights_dict, strict=False))

    model_featherFusion = create_Fusion(num_classes=args.num_classes,model_convNeXt=model_convNeXt,model_textCNN=model_textCNN,model_GCN=model_GCN)

    model_featherFusion.to(device)

    # pg = [p for p in model.parameters() if p.requires_grad]
    pg = get_params_groups(model_featherFusion, weight_decay=args.wd)
    # optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=args.wd)
    optimizer = optim.Adam(pg, lr=args.lr, weight_decay=args.wd)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs,
                                       warmup=True, warmup_epochs=1)

    best_acc = 0.
    best_pre = 0.
    best_rec = 0.
    best_f1 = 0.
    for epoch in range(args.epochs):
        print(" Epoch:",epoch,"Time:",time.ctime())
        # train
        torch.cuda.empty_cache()
        train_loss, train_acc = train_one_epoch(model=model_featherFusion,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch,
                                                lr_scheduler=lr_scheduler)
                                                # lr_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=64))

        # validate
        val_loss, val_acc,ACC,PRE,REC,F1 = evaluate(model=model_featherFusion,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)


        print("train_loss, train_acc:",train_loss, train_acc)
        print("val_loss, val_acc:",val_loss, val_acc)
        
        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        if best_acc < val_acc:
            torch.save(model_featherFusion.state_dict(), "apkPreprocess/network_all/convNeXt_weights/best_model60%.pth")
            best_acc = val_acc
            best_pre = PRE
            best_rec = REC
            best_f1 = F1
        print("best_acc,best_pre,best_rec,best_f1:",best_acc,best_pre,best_rec,best_f1)
    return best_acc,best_pre,best_rec,best_f1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=3e-6)
    parser.add_argument('--wd', type=float, default=0)
    # 数据集所在根目录
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str, default="apkPreprocess/TrainData")

    # 预训练权重路径，如果不想载入就设置为空字符
    # 链接: https://pan.baidu.com/s/1aNqQW4n_RrUlWUBNlaJRHA  密码: i83t
    parser.add_argument('--weights', type=str, default='apkPreprocess/network_all/convnext_tiny_1k_224_ema.pth',
                        help='initial weights path')
    # 是否冻结head以外所有权重
    # parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:1', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    sum_acc=0.
    sum_pre=0.
    sum_rec=0.
    sum_f1=0.

    print('epochs:{}, batch-size:{}, lr:{}, wd:{}, device:{}\n'.format(opt.epochs,opt.batch_size,opt.lr,opt.wd,opt.device))

    K=5
    for i in range(1):
    # for i in range(K):
        print(" Fold:",i,"Time:",time.ctime())
        best_acc,best_pre,best_rec,best_f1=train_1fold(opt,i,K)
        print("best_acc,best_pre,best_rec,best_f1:",best_acc,best_pre,best_rec,best_f1)
        # with open("apkPreprocess/network_all/bestRes.log", "a") as f:
        #     print("best_acc,best_pre,best_rec,best_f1:",best_acc,best_pre,best_rec,best_f1,flush=True,file=f)
        # sum_acc+=best_acc
        # sum_pre+=best_pre
        # sum_rec+=best_rec
        # sum_f1+=best_f1
    # avg_acc=sum_acc/K
    # avg_pre=sum_pre/K
    # avg_rec=sum_rec/K
    # avg_f1=sum_f1/K

    # print("avg_acc=",avg_acc,"avg_pre=",avg_pre,"avg_rec=",avg_rec,"avg_f1=",avg_f1)

    # print("avg_acc=",avg_acc,"avg_pre=",avg_pre,"avg_rec=",avg_rec,"avg_f1=",avg_f1)
    # with open("apkPreprocess/network_all/avgRes.log", "a") as f:
    #     print("avg_acc=",avg_acc,"avg_pre=",avg_pre,"avg_rec=",avg_rec,"avg_f1=",avg_f1,flush=True,file=f)