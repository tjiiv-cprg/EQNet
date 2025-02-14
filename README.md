# EQNet: Boost Efficiency for Query-Centric Multi-Agent Motion Forecasting

![image-20241206145648021](https://raw.githubusercontent.com/tjyuyao/imgurl/main/image-20241206145648021.png)

## SETUP

### PREPARE DATASET

- You need to first download Argoverse 2 motion forecasting dataset.
- You need to modify the data root in `EQNet/env_config.yml:machines/default/argoverse_v2_dataroot` to the correct location.
- The dataset folder structure example:
  ```
  /mnt/data/argoverse2/
  ├── test
  │   ├── processed
  │   └── raw
  ├── train
  │   ├── processed
  │   └── raw
  └── val
      ├── processed
      │   ├── 6a24bd-1ad8-44c7-9a0e-5e93bb41d5ae.pkl
      |   ...
      └── raw
          ├── 6a24bd-1ad8-44c7-9a0e-5e93bb41d5ae
          ...
  ```

  The `processed` folder contains the cache data, which is generated from the raw data. It will be automatically generated on the initial run. The `raw` folder contains the original data.

### PREPARE ENVIRONMENT

```bash
git clone <this-repo>

# we use conda to manage system packages
# to compile cuda extensions: gcc=9.5.0 gxx=9.5.0
# evalai dependencies: lxml
# av2-api dependencies: rust
conda create --name py310rust python=3.10 gcc=9.5.0 gxx=9.5.0 lxml rust virtualenv -c conda-forge 
conda activate py310rust
pip install virtualenv  # we use virtualenv to manage python packages

cd <repo_root>
python3 -m venv .venv  # to be able to navigate the library code in the explorer tree, create the .venv folder locally
source .venv/bin/activate
pip install -U setuptools pip

<install-pytorch-via-pip-with-proper-version-of-cuda>  # accord with /usr/local/cuda version
<install-pytorch-geometric-via-pip-with-proper-version-of-cuda>  # https://pytorch-geometric.readthedocs.io/en/1.6.3/notes/installation.html

pip install https://github.com/argoverse/av2-api.git  # mkdir .venv/src ; cd .venv/src; git clone git@github.com:argoverse/av2-api.git; pip install -e av2-api; cd ../..
pip install https://github.com/JonathonLuiten/TrackEval.git  # mkdir .venv/src ; cd .venv/src; git clone git@github.com:JonathonLuiten/TrackEval.git; pip install -e TrackEval; cd ../..
pip install beautifulsoup4==4.7.1 beautifultable==0.7.0 "boto3>=1.9.88" docker==3.6.0 validators==0.12.6 termcolor==1.1.0 "tqdm>=4.49.0"  # manually manage dependencies for evalai to avoid conflicts
pip install evalai --no-deps

pip install lightning einops torchscale huggingface_hub tensorboard==2.12.1

pip install GitPython json5 shapely descartes imageio omegaconf colour pyyaml docstring_parser typing_inspect ipdb matplotlib click

pip install "numpy<2.0"
```

## TRAINING

```
conda activate py310rust
source .venv/bin/activate
bash EQNet/train.sh -m
```

## TESTING

TBA

## BENCHMARK SPEED

TBA

## VISUALIZATION

- checkout `visualize.ipynb`.

## CITATION

```
TBA
```
