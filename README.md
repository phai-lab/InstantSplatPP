<h2 align="center">
  <a href="https://arxiv.org/abs/2403.20309">InstantSplat++: Sparse-view Gaussian Splatting in Seconds</a>
</h2>

<h5 align="center">

[![arXiv](https://img.shields.io/badge/Arxiv-2403.20309-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2403.20309)
[![Gradio](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/kairunwen/InstantSplat)
[![Home Page](https://img.shields.io/badge/Project-Website-green.svg)](https://instantsplat.github.io/)
[![X](https://img.shields.io/badge/-Twitter@Zhiwen%20Fan%20-black?logo=twitter&logoColor=1D9BF0)](https://x.com/WayneINR/status/1774625288434995219)
[![YouTube](https://img.shields.io/badge/Demo_Video-E33122?logo=Youtube)](https://youtu.be/fxf_ypd7eD8)
[![YouTube](https://img.shields.io/badge/Tutorial_Video-E33122?logo=Youtube)](https://www.youtube.com/watch?v=JdfrG89iPOA&t=347s)

</h5>

<div align="center">

This repository contains <b>InstantSplat++</b>, an improved extension of
<a href="https://github.com/NVlabs/InstantSplat">InstantSplat</a>
for sparse-view large-scale scene reconstruction with Gaussian Splatting.

InstantSplat++ preserves the original InstantSplat design and supports 3D-GS, 2D-GS, and Mip-Splatting.

<br/>

If you use this repository in your research, please <b>also cite the original InstantSplat paper and codebase</b>:
<a href="https://github.com/NVlabs/InstantSplat">NVlabs/InstantSplat</a>.

</div>

<br/>

## Table of Contents

- [Free-view Rendering](#free-view-rendering)
- [Get Started](#get-started)
  - [Installation](#installation)
  - [Usage](#usage)
- [Acknowledgement](#acknowledgement)
- [Citation](#citation)

## Free-view Rendering



---

## Get Started

InstantSplat++ is built on top of the original InstantSplat codebase.  
This guide provides a reproducible **conda setup** (recommended).

---

## Installation

### 1) Clone the repository + download the pre-trained model

```bash
git clone --recursive https://github.com/phai-lab/InstantSplatPP.git
cd InstantSplatPP

mkdir -p mast3r/checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth \
  -P mast3r/checkpoints/
````

### 2) Create conda environment

```bash
conda create -n instantsplatPP python=3.10.13 cmake=3.14.0 -y
conda activate instantsplatPP

# PyTorch + CUDA runtime (12.1)
conda install pytorch==2.1.2 torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y
python -m pip install -U pip
```

### 3) Install Python dependencies (keep NumPy < 2)

> NOTE: We pin `numpy<2` to avoid ABI issues with compiled CUDA extensions.

```bash
pip install -r requirements.txt

python -m pip uninstall -y numpy opencv-python
python -m pip install --no-cache-dir "numpy<2" "opencv-python<4.12"
```

### 4) Build & install CUDA submodules

```bash
pip install -v --no-build-isolation ./submodules/simple-knn
pip install -v --no-build-isolation ./submodules/diff-gaussian-rasterization
pip install -v --no-build-isolation ./submodules/fused-ssim
```

### 5) (Optional) Compile RoPE CUDA kernels (CroCo v2)

```bash
cd croco/models/curope/
python setup.py build_ext --inplace
cd ../../../
```

### 6) (Optional) MapAnything prior (third_party)

If you want to run with `PRIOR_MODEL_TYPE=mapanything`, install MapAnything from `third_party/`.
MapAnything may change dependencies (e.g. upgrade NumPy), so we install it with constraints and then rebuild CUDA submodules.

```bash
# constraints: keep NumPy < 2 and OpenCV compatible
cat > constraints.txt << 'EOF'
numpy<2
opencv-python<4.12
EOF

# install from local repo (already under third_party/)
pip install -e third_party/mapanything -c constraints.txt

# enforce NumPy<2 (in case it was upgraded), then rebuild CUDA submodules
python -m pip install --no-cache-dir "numpy<2"

rm -rf submodules/simple-knn/build \
       submodules/diff-gaussian-rasterization/build \
       submodules/fused-ssim/build

pip install -v --no-build-isolation ./submodules/simple-knn
pip install -v --no-build-isolation ./submodules/diff-gaussian-rasterization
pip install -v --no-build-isolation ./submodules/fused-ssim
```

---

## Usage

### 1) Data preparation

Download our pre-processed data from:

* [Hugging Face](https://huggingface.co/datasets/kairunwen/InstantSplat)
* [Google Drive](https://drive.google.com/file/d/1Z17tIgufz7-eZ-W0md_jUlxq89CD1e5s/view)

Place your data under `assets/examples/<scene_name>/images` (or follow the same folder structure).

### 2) Commands

```bash
# Train + render (no GT reference, interpolate camera trajectory)
bash scripts/run_infer.sh

# Train + evaluate (with GT reference)
bash scripts/run_eval.sh

# Run with prior models (e.g., VGGT / MapAnything)
bash scripts/run_all_prior_model.bash
```

---

## Acknowledgement

This work is built on many amazing research works and open-source projects. Thanks to all the authors for sharing!

* [InstantSplat (original framework)](https://github.com/NVlabs/InstantSplat)
* [Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) and [diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization)
* [DUSt3R](https://github.com/naver/dust3r)
* [MASt3R](https://github.com/naver/mast3r)
* [MapAnything](https://github.com/facebookresearch/map-anything)
* [VGGT](https://github.com/facebookresearch/vggt)

---

## Citation

If you find our work useful, please consider giving a star ⭐ and citing:

```bibTeX
@misc{fan2025instantsplatsparseviewgaussiansplatting,
      title={InstantSplat: Sparse-view Gaussian Splatting in Seconds},
      author={Zhiwen Fan and Wenyan Cong and Kairun Wen and Kevin Wang and Jian Zhang and Xinghao Ding and Danfei Xu and Boris Ivanovic and Marco Pavone and Georgios Pavlakos and Zhangyang Wang and Yue Wang},
      year={2025},
      eprint={2403.20309},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2403.20309},
}
```

> Note: InstantSplat++ is an extension of the original **InstantSplat** framework.
> Please cite the paper above and acknowledge the original codebase: [https://github.com/NVlabs/InstantSplat](https://github.com/NVlabs/InstantSplat)

