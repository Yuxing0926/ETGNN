# ETGNN
A General Tensor Prediction Framework Based on Graph Neural Networks



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


## Code contributors:
+ Yuxing Ma (Fudan University)
+ Yang Zhong (Fudan University)

## References
The papers related to HamGNN:
[[1] Transferable equivariant graph neural networks for the Hamiltonians of molecules and solids](https://doi.org/10.1038/s41524-023-01130-4)
