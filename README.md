<p align="center">
<h1 align="center">PIPE Planner: Pathwise Information Gain with Map Predictions for Indoor Robot Exploration</h1>
<h3 class="is-size-5 has-text-weight-bold" style="color: orange;" align="center">
    IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) 2025
</h3>
  <p align="center">
    <a href="https://seungjaebk.github.io/" target="_blank"><strong>Seungjae Baek*</strong></a>
    ·
    <a href="https://bradymoon.com/" target="_blank"><strong>Brady Moon*</strong></a>
    ·
    <a href="https://seungchan-kim.github.io" target="_blank"><strong>Seungchan Kim*</strong></a>
     <br>
    <a href="https://caomuqing.github.io/" target="_blank"><strong>Muqing Cao</strong></a>
    ·
    <a href="https://cherieho.com/" target="_blank"><strong>Cherie Ho</strong></a>
    ·
    <a href="https://theairlab.org/team/sebastian/" target="_blank"><strong>Sebastian Scherer</strong></a>
    ·
    <a href="https://rml-unist.notion.site/" target="_blank"><strong>Jeong hwan Jeon</strong></a>
    <br>
  </p>
</p>
  <h3 align="center"><a href="https://arxiv.org/abs/2503.07504">Paper</a> | <a href="https://pipe-planner.github.io">Project Page</a> | <a href="https://youtu.be/oZEqbCBRn-I">Video</a></h3>
  <div align="center"></div>

## Preliminary Setup
### Clone the github repository
Clone the repository as below.

    git clone --branch init-import --single-branch https://github.com/castacks/pipe-planner.git
    cd pipe-planner

### Set up Conda Environment
Create environment with the name 'pipe' from lama's conda_env.yml
    
    conda env create -n pipe -f lama/conda_env.yml
    conda activate pipe

#### Check "range_libc" Already Installed

    python -c "import range_libc; print('range_libc installed successfully')"


#### If Not Installed, Build from Source

    cd range_libc/pywrapper
    
    # Install build dependencies (if needed)
    conda install -y cython
    
    # Build and install
    python setup.py install
    
    # Verify installation
    cd ../..
    python -c "import range_libc; print('range_libc installed successfully')"

### Download pretrained prediction models (KTH dataset)
You can download pretrained models from this <a href="https://drive.google.com/drive/u/0/folders/1u9WZ9ftwaMbP-RVySuNSVEdUDV_x4Dw6">link</a>. Place the zip file under `pretrained_models` directory and unzip the file. 

    mv ~/Downloads/weights.zip ~/pipe-planner/pretrained_models/
    cd ~/pipe-planner/pretrained_models/
    unzip weights.zip

The `pretrained_model` directory and its subdirectories should be organized as below: 

    pipe-planner
    ├── pretrained_models
        ├── weights
            ├── big_lama
                ├── models
                    ├── best.ckpt
            ├── lama_ensemble
                ├── train_1
                    ├── models
                        ├── best.ckpt
                ├── train_2
                    ├── models
                        ├── best.ckpt
                ├── train_3
                    ├── models
                        ├── best.ckpt    

## Experiments
### Customize your own experiment
In configs/base.yaml, you can manually select map, starting pose, and planning method. All map information and starting points available at <a href="https://magenta-brow-f14.notion.site/25-Starting-Points-per-Map-28d544fc91ed80c5bbdbdc1fb49a13de?pvs=143">here</a>.


#### log_iou 
If true, your algorithm runs until reaching the maximum time step budget (1500 for small maps, 3000 for medium maps, and 6000 for large maps), or reaching the 95% IoU. It saves the IoU score per 20 time steps and when reaching 90% and 95% IoU. If false, the algorithms for designated 'mission_time' time step.

### Run the script
Run the 'explore.py' script as below:

    cd scripts/
    python3 explore.py


## Citation

If you find our paper or code useful, please cite us:

```bib
@article{baek2025pipe,
  title={PIPE Planner: Pathwise Information Gain with Map Predictions for Indoor Robot Exploration},
  author={Baek, Seungjae and Moon, Brady and Kim, Seungchan and Cao, Muqing and Ho, Cherie and Scherer, Sebastian and others},
  journal={arXiv preprint arXiv:2503.07504},
  year={2025}
}
