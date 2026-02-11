import argparse
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import torch


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = Unet(
	dim = 64,
	dim_mults = (1, 2, 4, 8)
).to(device=DEVICE)


diffusion = GaussianDiffusion(
	model,
	lc_size = 256,
	timesteps = 1000,
	loss_type = 'l1'
).to(device=DEVICE)


parser = argparse.ArgumentParser("")
parser.add_argument('--milestone', default=0, dest='milestone', type=int, help='start at this number')
parser.add_argument('--data_dir', default='/media/bigdata/PRIMVS/light_curves/', type=str, help='PRIMVS data directory')
parser.add_argument('--fits_file', default=None, type=str, help='FITS file with source IDs (optional)')
parser.add_argument('--band', default='Ks', type=str, help='Photometric band to use')
args = parser.parse_args()


trainer = Trainer(
	diffusion,
	args.data_dir,
	fits_file = args.fits_file,
	band = args.band,
	logdir = './logs/',
	lc_size = 70,
	train_batch_size = 56,
	train_lr = 2e-5,
	train_num_steps = 750001,		 # total training steps
	gradient_accumulate_every = 2,	# gradient accumulation steps
	ema_decay = 0.995,				# exponential moving average decay
	num_workers=32,
	rank = [0, 1, 2]
)

if args.milestone != 0:
	trainer.load(args.milestone)
trainer.train()
