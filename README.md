# ETGNN
A General Tensor Prediction Framework Based on Graph Neural Networks

## Requirements
We recommend using Python 3.9. ETGNN requires the following Python libraries:
- `numpy`
- `PyTorch`
- `PyTorch Geometric`
- `pytorch_lightning`
- `e3nn`
- `pymatgen`
- `tensorboard`
- `tqdm`
- `scipy`
- `yaml`
- You can directly use the provided 'environment.yaml' to set up the environment.


## Usage
### Preparation of Hamiltonian Training Data
1. **Generate Structure Files**: Create structure files (e.g., POSCAR or CIF) via molecular dynamics or random perturbation.
2. **Construct Tensor Labels files**: Create an 'id_prop.csv' file to store the corresponding tensor labels for each structure, which will be used for training.
   
### Network Training and prediciton
1. **Configure the Network**: Set the appropriate parameters in the `config.yaml` file for network and training configurations.
2. **Train network**: Run the training process with:`python .../main.py` in the working directory.
3. **Monitor Training**: Use TensorBoard to track training progress: `tensorboard --logdir train_dir`
4. **Prediction**: After completing the training process, the model can be used to make predictions. Follow these steps:
- Set `checkpoint_path` in `config.yaml` to the trained model's path and `stage` to `test`.
- -Run `python .../main.py` in the predciton directory.


## Code contributors:
+ Yuxing Ma (Fudan University)
+ Yang Zhong (Fudan University)

## References
The papers related to HamGNN:
[[1] A General Tensor Prediction Framework Based on Graph Neural Networks](https://doi.org/10.1021/acs.jpclett.3c01200)
