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
- You can directly use the provided `environment.yaml` to set up the environment.

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
- Run `python .../main.py` in the predciton directory.
##  Explanation of the parameters in config.yaml
The input parameters in config.yaml are divided into different modules, which mainly include `'setup'`, `'dataset_params'`, `'losses_metrics'`, `'optim_params'` and network-related parameters. Most of the parameters work well using the default values. The following introduces some commonly used parameters in each module (The parameter settings in the network-related modules are omitted.).
+ `setup`:
    + `GNN_Net`<EF><BC><9A>Choose an available network in LightGNN to calculate the representation of crystals or molecules, such as 'CGCNN', 'SchNet', 'DimeNet_Triplet'.
    + `stage`: Select the state of the network: training ('fit') or testing ('test').
    + `property`<EF><BC><9A>Select the type of physical quantity to be output by the network: scalar ('scalar'), atomic average scalar ('scalar_per_atom'), Born effective charge ('Born'), force ('force'), dielectric ('dielectric').
    + `l_pred_atomwise_tensor`: Set to True to predict tensors for each atom.
    + `l_pred_crystal_tensor`<EF><BC><9A>Set to True to predict the tensor properties of the crystal.
    + `num_gpus`: number of gpus to train on (int) or which GPUs to train on (list or str) applied per node.
    + `resume`: resume training ('true') or start from scratch ('false').
    + `checkpoint_path`: Path of the checkpoint from which training is resumed (`stage` = 'fit') or path to the checkpoint you wish to test (`stage` = 'test').
+ `dataset_params`<EF><BC><9A>
    + `crystal_path`: Directory for storing crystal structure files.
    + `file_type`: The type of crystal structure, cif file ('cif') or POSCAR format file ('poscar').
    + `graph_data_path`: The directory where the processed compressed graph data files are stored. If there is already a file named 'grah_data.npz' in this directory, the program will directly read this file to construct the dataset.
    + `id_prop_path`: The path of the id_prop.csv file that saves the id and property of each crystal structure.
    + `max_num_nbr`: The maximum number of neighbors of each node when generating graph data of the crystal structures.
    + `radius`: Maximum cutoff radius of each node.
    + `rank_tensor`: The order of the tensor to be predicted, 0 represents a scalar, 1 represents a first-order vector, and 2 represents a second-order tensor.
    + `train_ratio`: The proportion of the training samples in the entire data set.
    + `val_ratio`: The proportion of the validation samples in the entire data set.
    + `test_ratio`<EF><BC><9A>The proportion of the test samples in the entire data set.
+ `losses_metrics`<EF><BC><9A>
    + `losses`: define multiple loss functions and their respective weights in the total loss value. Currently, LightGNN supports mse, mae, cosine_similarity, sum_zero and Euclidean_loss.
    + `metrics`<EF><BC><9A>A variety of metric functions can be defined to evaluate the accuracy of the model on the validation set and test set.
+ `optim_params`<EF><BC><9A>
    + `min_epochs`: Force training for at least these many epochs.
    + `max_epochs`: Stop training once this number of epochs is reached.
    + `lr`<EF><BC><9A>learning rate, the default value is 0.0005.
+ `profiler_params`:
    + `train_dir`: The folder for saving training information and prediction results. This directory can be read by tensorboard to monitor the training process.

## Code contributors:
+ Yuxing Ma (Fudan University)
+ Yang Zhong (Fudan University)

## References
The papers related to HamGNN:
[[1] A General Tensor Prediction Framework Based on Graph Neural Networks](https://doi.org/10.1021/acs.jpclett.3c01200)

## Appendix: Expressions for tensor output</font>
<font face="Times New Roman" size=4>&emsp;one-order tensor:</font>
$$\vec{F_j}=\sum_{k\in{N(j)}} E_{kj} \vec{e_{kj}}$$
<font face="Times New Roman" size=4>&emsp;two-order tensor:</font>
$$T_{j}=\sum_{k\in{N(j)}} E_{kj}\vec{e_{kj}} \bigotimes \vec{e_{kj}} + \sum_{k,i\in{N(j)}} E_{kji}\vec{e_{kj}} \bigotimes \vec{e_{ji}}$$

