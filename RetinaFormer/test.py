import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim


from torch.utils.data import DataLoader
from collections import OrderedDict
import pynvml

from utils import AverageMeter, write_img, chw_to_hwc
from datasets.loader import PairLoader
from models import retinaFormer #model
from models.retinaFormer import retinaformer_m

from torchstat import stat
from torchinfo import summary

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='retinaformer_m', type=str, help='model name')
parser.add_argument('--num_workers', default=16, type=int, help='number of workers')
parser.add_argument('--data_dir', default='./data/', type=str, help='path to dataset')
parser.add_argument('--save_dir', default='./saved_models/', type=str, help='path to models saving')
parser.add_argument('--result_dir', default='./results/', type=str, help='path to results saving')
parser.add_argument('--dataset', default='ITS', type=str, help='dataset name')
parser.add_argument('--exp', default='indoor', type=str, help='experiment setting')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 例如 '0' 或 '1'


def single(save_dir):
	state_dict = torch.load(save_dir)['state_dict']
	new_state_dict = OrderedDict()

	for k, v in state_dict.items():
		name = k[7:]
		new_state_dict[name] = v

	return new_state_dict


def test(test_loader, network, result_dir):
	PSNR = AverageMeter()
	SSIM = AverageMeter()

	torch.cuda.empty_cache()

	network.eval()

	os.makedirs(os.path.join(result_dir, 'imgs'), exist_ok=True)
	f_result = open(os.path.join(result_dir, 'results.csv'), 'w')

	for idx, batch in enumerate(test_loader):
		input = batch['source'].cuda()
		target = batch['target'].cuda()

		filename = batch['filename'][0]

		with torch.no_grad():
			output = network(input)
			output = output.clamp_(-1, 1)

			output = output * 0.5 + 0.5
			target = target * 0.5 + 0.5
			#psnr TIP2023 same
			mse_loss = F.mse_loss(output, target, reduction='none').mean((1, 2, 3))
			psnr_val = 10 * torch.log10(1 / mse_loss).mean().item()

			# # # TIP 2023 下采样 偏高
			# _, _, H, W = output.size()
			# down_ratio = max(1, round(min(H, W) / 256))		# Zhou Wang
			# ssim_val = ssim(F.adaptive_avg_pool2d(output, (int(H / down_ratio), int(W / down_ratio))),
			# 				F.adaptive_avg_pool2d(target, (int(H / down_ratio), int(W / down_ratio))),
			# 				data_range=1, size_average=False).item()

			#ICCV 2023
			ssim_val = ssim(output,
						target,
						data_range=1.0, size_average=True).item()
			#CVPR 2024
			#ssim_val = ssim(output[0].permute(1, 2, 0).cpu().numpy(), target[0].permute(1, 2, 0).cpu().numpy(), channel_axis=2, data_range=1)

		PSNR.update(psnr_val)
		SSIM.update(ssim_val)

		print(f'Test: [{idx}]\t'
			  f'name: [{filename}]\t'
			  f'PSNR: {PSNR.val:.02f} ({PSNR.avg:.02f})\t'
			  f'SSIM: {SSIM.val:.03f} ({SSIM.avg:.03f})')

		with open(os.path.join(args.result_dir, 'psnr_ssim.txt'), 'a') as f:
			f.write(filename + '.PNG ---->' + "PSNR: %.4f, SSIM: %.4f] " % (psnr_val, ssim_val) + '\n')

		f_result.write('%s,%.02f,%.03f\n'%(filename, psnr_val, ssim_val))
		#保存图片
		out_img = chw_to_hwc(output.detach().cpu().squeeze(0).numpy())
		#write_img(os.path.join(result_dir, 'imgs', 'input'+filename), chw_to_hwc(input.detach().cpu().squeeze(0).numpy()))
		write_img(os.path.join(result_dir, 'imgs', filename), out_img)
		# B_img = chw_to_hwc(B.detach().cpu().squeeze(0).numpy())
		# write_img(os.path.join(result_dir, 'B_imgs', filename), B_img)
		# L_img = chw_to_hwc(L.detach().cpu().squeeze(0).numpy())
		# write_img(os.path.join(result_dir, 'L_imgs', filename), L_img)

	f_result.close()

	os.rename(os.path.join(result_dir, 'results.csv'),
			  os.path.join(result_dir, '%.02f | %.04f.csv'%(PSNR.avg, SSIM.avg)))

def print_info(model,input):
	if torch.is_tensor(input):
		if input.dim()>3:
			print('summary model info')
			summary(model,input.shape)
			# input = input[1,:,:,:].cuda()
			#打印内存使用率
			pynvml.nvmlInit()
			handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # 0表示第一块显卡
			meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
			ava_mem = round(meminfo.free / 1024 ** 2)
			print('current available video memory is' + ' : ' + str(round(meminfo.free / 1024 ** 2)) + ' MIB')
	else:
		print('输入信息错误')

if __name__ == '__main__':
	network = eval(args.model.replace('-', '_'))()
	network.cuda()
	saved_model_dir = os.path.join(args.save_dir, args.exp, args.model+'itsm0.5caE30041.75.pth')

	if os.path.exists(saved_model_dir):
		print('==> Start testing, current model name: ' + args.model)
		network.load_state_dict(single(saved_model_dir))
	else:
		print('==> No existing trained model!')
		exit(0)
	"""
	参数计算
	input_x = torch.Tensor(16,3,256,256).cuda()
	print_info(network, input_x)
	"""
	dataset_dir = "/home/ubuntu/zhangyun/code/onoffFormer/data/ITS"
	test_dataset = PairLoader(dataset_dir, 'test', 'test')
	test_loader = DataLoader(test_dataset,
							 batch_size=1,
							 num_workers=args.num_workers,
							 pin_memory=True)

	result_dir = os.path.join(args.result_dir, args.dataset, args.model)

	test(test_loader, network, result_dir)