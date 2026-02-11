
# LC_DDPM (Light Curve Denoising Diffusion Probabilistic Models)

This repository contains a PyTorch implementation of **Denoising Diffusion Probabilistic Models (DDPM)** specifically adapted for generating synthetic astronomical light curves. The model uses a UNet architecture to learn the distribution of periodic variable light curves and generate new, realistic samples.

## Features

* **Custom Diffusion Model:** Implementation of Gaussian Diffusion adapted for 1D/sequential light curve data using 2D convolutions (treating time/phase series similarly to image channels/dimensions).
* **Stochastic Data Loading:** Includes a custom `StochasticCurves` dataset loader designed to handle light curve CSVs, including magnitude, error, phase, and time.
* **HPC Ready:** Includes scripts (`DDPM.sh`) for submitting training jobs to PBS/Torque-based GPU clusters.
* **Inference Pipeline:** Scripts to generate (sample) new light curves from trained model checkpoints.

## File Structure

* `denoising_diffusion_pytorch.py`: The core library containing the `Unet` architecture, `GaussianDiffusion` logic, `Trainer` class, and data loading utilities (`Lightcurves`, `StochasticCurves`).
* `train.py`: The entry point for training the model.
* `infer.py`: The entry point for generating (sampling) data using a trained checkpoint.
* `DDPM.sh`: A PBS shell script for submitting training jobs to a GPU cluster.

## Requirements

To run this code, you will need the following Python libraries:

```txt
torch
numpy
astropy
tqdm
einops
torchvision

```

## Configuration (Important)

**Note on Data Paths:**
The scripts currently contain hardcoded absolute paths specific to the original training environment (e.g., `/beegfs/car/njm/...`). Before running, you **must** update these paths to point to your local data directories.

1. **In `denoising_diffusion_pytorch.py**`:
* Update the `sequences_pickle` path in the `Trainer.__init__` method.
* Update the `tier` variable (glob pattern) to point to your CSV files.


2. **In `train.py**`:
* Update the path in the `Trainer` initialization: `'/beegfs/car/njm/Periodic_Variables/LC/'`.


3. **In `DDPM.sh**`:
* Update the python execution path if not running on the specific cluster originally designed for.



## Usage

### 1. Data Preparation

The model expects data in CSV format. The `StochasticCurves` class reads files expecting columns for Magnitude, Error, Phase, and Time. Ensure your data matches the format expected by the `np.genfromtxt` call in `denoising_diffusion_pytorch.py`.

### 2. Training

To train the model, run `train.py`. You can specify a milestone (checkpoint step) if resuming training.

```bash
# Start from scratch
python train.py

# Resume from a specific checkpoint (e.g., step 177000)
python train.py --milestone 177000

```

**Training Parameters:**
You can adjust hyperparameters inside `train.py`:

* `lc_size`: Length of the light curve sequence.
* `train_batch_size`: Batch size.
* `train_lr`: Learning rate.
* `train_num_steps`: Total training steps.

### 3. Inference (Generation)

To generate synthetic light curves using a trained model, use `infer.py`.

```bash
python infer.py --dataset probes --milestone 750000 --batches 10

```

**Arguments:**

* `--dataset`: Choices are `probes` or `sdss`. (Ensure you configure the paths for these choices inside `infer.py`).
* `--milestone`: The training step number of the model checkpoint you wish to load.
* `--batches`: The number of batches to generate.

Output files are saved as `.npy` files in the `inferred/` directory.

### 4. HPC Submission

If you are using a cluster with a PBS scheduler, you can submit the job using the provided shell script:

```bash
qsub DDPM.sh

```

## Model Architecture

The model utilizes a UNet with:

* Sinusoidal Positional Embeddings.
* ResNet blocks with Mish activation.
* Linear Attention mechanisms in the bottleneck and up/down-sampling blocks.
* Gaussian Diffusion process with a cosine beta schedule.
