import os
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.ao.quantization.fx.prepare import _save_state
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
from pytorch_msssim import ssim, ms_ssim
from utils import AverageMeter
from utils.crloss import ContrastLoss
from datasets.loader import PairLoader
from models import retinaFormer #model
from models.retinaFormer import retinaformer_m


# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='retinaformer-m', type=str, help='model name')  # 选用的模型
parser.add_argument('--json', default='retinaformer-m', type=str, help='model name')  # 选用的模型
parser.add_argument('--num_workers', default=16, type=int, help='number of workers')  # 数据加载的线程数
parser.add_argument('--no_autocast', action='store_false', default=True, help='disable autocast')  # 是否禁用自动混合精度
parser.add_argument('--save_dir', default='./saved_models/', type=str, help='path to models saving')  # 模型保存路径
parser.add_argument('--data_dir', default='./data/', type=str, help='path to dataset')  # 数据集路径
parser.add_argument('--log_dir', default='./logs/', type=str, help='path to logs')  # 日志保存路径
parser.add_argument('--dataset', default='ITS', type=str, help='dataset name')  # 数据集名称
parser.add_argument('--exp', default='indoor', type=str, help='experiment setting')  # 实验设置
parser.add_argument('--gpu', default='0', type=str, help='GPUs used for training')  # 使用的GPU
parser.add_argument('--result_dir', default='./results/', type=str, help='path to results saving')  # 结果保存路径
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu  # 例如 '0' 或 '1'

# CR loss
CRloss = ContrastLoss(ablation=False)
def train(train_loader, network, criterion, optimizer, scaler):
    losses = AverageMeter()
    network.train()

    for batch in train_loader:
        source_img = batch['source'].cuda()
        target_img = batch['target'].cuda()

        optimizer.zero_grad()

        with autocast(args.no_autocast):
            # 修改点1：移除原地操作
            output = network(source_img)
            output_clamped = output.clamp(-1, 1)  # 非原地操作
            output_normalized = output_clamped * 0.5 + 0.5
            target_normalized = target_img * 0.5 + 0.5
            # L1 loss
            lossL1 = criterion(output_clamped, target_img)  # 使用clamped输出计算L1
            # perceptual_loss
            # perceptual_loss = loss_network(output_normalized, target_normalized)
            # CR loss
            loss_CR = CRloss(output_clamped, target_img, source_img)
            loss = lossL1 + 0.3* loss_CR

        # 修改点2：调整backward顺序
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        losses.update(loss.item())

    return losses.avg


def valid(val_loader, network):
    PSNR = AverageMeter()  # 用于记录 PSNR

    torch.cuda.empty_cache()  # 释放未被分配的显存

    network.eval()  # 设置为评估模式

    for batch in val_loader:
        source_img = batch['source'].cuda()
        target_img = batch['target'].cuda()

        with torch.no_grad():  # 禁用梯度计算
            output = network(source_img).clamp_(-1, 1)  # 限制输出范围

        # 计算 MSE Loss 和 PSNR
        mse_loss = F.mse_loss(output * 0.5 + 0.5, target_img * 0.5 + 0.5, reduction='none').mean((1, 2, 3))
        psnr = 10 * torch.log10(1 / mse_loss).mean()
        PSNR.update(psnr.item(), source_img.size(0))  # 更新 PSNR 记录

    return PSNR.avg  # 返回平均 PSNR


if __name__ == '__main__':
    # 加载配置文件
    setting_filename = os.path.join('configs', args.exp, args.json + '.json')
    if not os.path.exists(setting_filename):
        setting_filename = os.path.join('configs', args.exp, 'default.json')
    with open(setting_filename, 'r') as f:
        setting = json.load(f)

    # 初始化模型
    network = eval(args.model.replace('-', '_'))()
    network = nn.DataParallel(network).cuda()

    # 定义损失函数
    criterion = nn.L1Loss()

    # 定义优化器
    if setting['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(network.parameters(), lr=setting['lr'])
    elif setting['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(network.parameters(), lr=setting['lr'])
    else:
        raise Exception("ERROR: unsupported optimizer")

    # 定义学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=setting['epochs'],
                                                           eta_min=setting['lr'] * 1e-1)
    scaler = GradScaler()  # 初始化梯度缩放器

    # 加载数据集
    #dataset_dir = os.path.join(args.data_dir, args.dataset)
    dataset_dir = "/home/ubuntu/zhangyun/code/onoffFormer/data/ITS"
    train_dataset = PairLoader(dataset_dir, 'train', 'train',
                               setting['patch_size'], setting['edge_decay'], setting['only_h_flip'])
    train_loader = DataLoader(train_dataset,
                              batch_size=setting['batch_size'],
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              drop_last=True)
    val_dataset = PairLoader(dataset_dir, 'test', 'test',
                             setting['patch_size'])
    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            num_workers=args.num_workers,
                            pin_memory=True)

    # 创建保存目录
    save_dir = os.path.join(args.save_dir, args.exp)
    os.makedirs(save_dir, exist_ok=True)

    # 训练模型
    if not os.path.exists(os.path.join(save_dir, args.model + 'retinaformer-mits03caE500.pth')):
        print('==> Start training, current model name: ' + args.model)

        writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.exp, args.model))

        best_psnr = 0
        for epoch in tqdm(range(setting['epochs'] + 1)):
            loss = train(train_loader, network, criterion, optimizer, scaler)

            writer.add_scalar('train_loss', loss, epoch)  # 记录训练损失

            scheduler.step()  # 更新学习率

            if epoch % setting['eval_freq'] == 0:
                avg_psnr = valid(val_loader, network)

                writer.add_scalar('valid_psnr', avg_psnr, epoch)  # 记录验证 PSNR

                if avg_psnr > best_psnr:
                    best_psnr = avg_psnr
                    torch.save({'state_dict': network.state_dict()},
                               os.path.join(save_dir, args.model + 'its03caE500.pth'))  # 保存最佳模型

                writer.add_scalar('best_psnr', best_psnr, epoch)  # 记录最佳 PSNR
                print(f'epoch {epoch + 1}, loss {loss:.3f}, valid_psnr {avg_psnr:.3f}, best_psnr {best_psnr:.3f}')
    else:
        print('==> Existing trained model')

        writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.exp, args.model)) #读取模型
        save_state = torch.load(os.path.join(save_dir, args.model+'retinaformer-mits03caE500.pth'))
        network.load_state_dict(save_state['state_dict'])#取键值就用[]+‘’
        best_psnr = 0
        for epoch in tqdm(range(setting['epochs'] + 1)):
            loss = train(train_loader, network, criterion, optimizer, scaler)

            writer.add_scalar('train_loss', loss, epoch)

            scheduler.step()
            # torch.save({'state_dict': network.state_dict()},
            #            os.path.join(save_dir, args.model + 'oh2480.pth'))
            if epoch % setting['eval_freq'] == 0:
                avg_psnr = valid(val_loader, network)

                writer.add_scalar('valid_psnr', avg_psnr, epoch)  # 记录验证 PSNR

                if avg_psnr > best_psnr:
                    best_psnr = avg_psnr
                    torch.save({'state_dict': network.state_dict()},
                               os.path.join(save_dir, args.model + 'its704.pth'))  # 保存最佳模型

                writer.add_scalar('best_psnr', best_psnr, epoch)  # 记录最佳 PSNR
                print(f'epoch {epoch + 1}, loss {loss:.3f}, valid_psnr {avg_psnr:.3f},best_psnr {best_psnr:.3f}')

