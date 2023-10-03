
## Installation
The code is based on The code is based on  BrainCog and tianshou. 
BrainCog is an open source spiking neural network based brain-inspired 
cognitive intelligence engine for Brain-inspired Artificial Intelligence and brain simulation. More information on BrainCog can be found on its homepage http://www.brain-cog.network/;
Install 'tianshou' framework first https://github.com/thu-ml/tianshou

1. We recommend using conda to build the environment:

    conda create -n <env_name>

2. Install the dependent packages:

    pip install -r requirements.txt

3. Install the BrainCog:

   # Install locally
    Enter the folder of braincog

    > `cd Brain-Cog-dev`

   Install braincog locally

    > `pip install -e .`
 
## Requirements:
python == 3.8

CUDA toolkit == 11

numpy >= 1.21.2

scipy >= 1.8.0

h5py >= 3.6.0

torch >= 1.10

torchvision >= 0.12.0

torchaudio  >= 0.11.0

timm >= 0.5.4

matplotlib >= 3.5.1

einops >= 0.4.1

thop >= 0.0.31

pyyaml >= 6.0

loris >= 0.5.3

pandas >= 1.4.2

tonic 

xlrd == 1.2.0

scikit-learn

seaborn

pygame

dv

tensorboard

gym

atari-py

opencv-python

tianshou

## Train

Change the seventh line in file network.py from sys.path.insert(0,r'D:\Brain-Cog-main (2)\Brain-Cog-main\examples\decision_making\RL') to sys.path.insert(0,r'your project path\examples\decision_making\RL').

```shell  
python ./examples/decision_making/RL/MDS-FQF/python main.py
```

