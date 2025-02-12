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
2. **Convert to OpenMX Format**: Edit the `poscar2openmx.yaml` file with appropriate path settings and run:
    ```bash
    poscar2openmx --config path/to/poscar2openmx.yaml
    ```
   This converts the structures into OpenMX’s `.dat` format.
3. **Run OpenMX**: Perform static calculations on the structure files to generate `.scfout` binary files containing Hamiltonian and overlap matrix information.
4. **Process with openmx_postprocess**: Run `openmx_postprocess` to generate the `overlap.scfout` file, which contains the Hamiltonian matrix `H0`, independent of the self-consistent charge density.
### Network Training and prediciton
1. **Configure the Network**: Set the appropriate parameters in the `config.yaml` file for network and training configurations.
2. **Train network**: Run the training process with:'python .../main.py' in the working directory.
3. **Monitor Training**: Use TensorBoard to track training progress: 'tensorboard --logdir train_dir'
4. **Prediction**: After completing the training process, the model can be used to make predictions. Follow these steps:
- Set `checkpoint_path` in `config.yaml` to the trained model's path and `stage` to `test`.
- -Run 'python .../main.py' in the predciton directory.


## Code contributors:
+ Yuxing Ma (Fudan University)
+ Yang Zhong (Fudan University)

## References
The papers related to HamGNN:
[[1] Transferable equivariant graph neural networks for the Hamiltonians of molecules and solids](https://doi.org/10.1038/s41524-023-01130-4)
