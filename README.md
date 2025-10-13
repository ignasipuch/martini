# 🧬 Martini MN5

A lightweight Python toolkit to streamline **Martini coarse-grained molecular dynamics simulations** — from all-atom input structures to fully prepared Martini systems — designed especially for use on the **MN5 cluster**.

---

## 📘 Table of Contents

- [About](#about)
- [Features](#features)
- [Installation and Example](#installation-and-example)

---

## 🧩 About

**Martini MN5** automates and simplifies the process of preparing and running **Martini CG simulations**.  
It wraps together:

- All-atom (AA) to coarse-grained (CG) conversion  
- Solvation and box setup  
- Membrane insertion and ionization  
- Replica generation and simulation setup  

This library was developed to work seamlessly within the **MN5 cluster environment**, but can be easily adapted for other HPC clusters.

---

## 🚀 Features

- 🧱 Converts all-atom structures to Martini coarse-grained models  
- 💧 Builds and solvates simulation boxes (including membranes)  
- 🔬 Generates and parameterizes Martini simulations  
- ⚙️ Automatically prepares job directories and input files  
- 🧮 Supports multiple replicas and temperature control  
- 💡 Modular, extendable API  

---

## ⚙️ Installation and Example

Clone and install in editable mode:

```bash
git clone https://github.com/ignasipuch/martini.git
cd martini
pip install -e .
```

This is an example of its functions. Specific examples with pdbs can be found in docs/examples.
```
import martini
model = martini.Martini(pdb_file='model.pdb', project_name='model')

# Residues to create a vector to align with the z-axis across the bilayer
residues = [('A', 1), ('A', 10)]

# Coarse-graining the AA model
model.setProteinCGModel(
    strength_conf=100,
    strength_el=0.7,
    residue_orientation=residues,
    maxwarn=30,
    overwrite=True
)

# Creating the solvation box and solvating the model
box_dim = [22, 22, 18]
model.setSolventCGModel(
    ion_molarity=0.15,
    membrane=True,
    z_membrane_shift=-4,
    box_dimensions=box_dim
)

# Set up Martini simulation
model.setUpMartiniSimulation(
    temperature=309.15,
    replicas=4,
    simulation_time=8000
)
```
