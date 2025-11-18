# How to set up environment to run our code 
```bash
conda create -n rectflow python=3.10
conda activate rectflow
pip install seaborn
pip install lightgbm
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install tensorflow==2.9.0 tensorflow-probability==0.12.2 tensorflow-gan==2.0.0 tensorflow-datasets==4.6.0
pip install jax==0.3.4 jaxlib==0.3.2 -f https://storage.googleapis.com/jax-releases/jax_releases.html
pip install numpy==1.21.6 ninja==1.11.1 matplotlib==3.7.0 ml_collections==0.1.1 scikit-learn==1.3.2 pandas==2.0.3
pip install hyperimpute # on windows, you may have to use hyperimpute==0.1.1
python -m pip install git+https://github.com/treforevans/uci_datasets.git
```

Then run each jupyter notebook to see results. 

# Approximate Run Time

On a machine with 16 core CPU and Nvidia RTX 3090 GPU, with 64 GB RAM, 

- ```demo_toy``` takes about a few minutes 
- ```demo_UCI``` takes about about 5 mintues on the ```wine``` dataset
- ```demo_imgs``` takes about about 6 hours