# 2D TMDC Generator (mcmaxent_2d.py)

This Python script uses the Atomic Simulation Environment (ASE) to build 2D Transition Metal Dichalcogenide (MX2) supercells using Maximum Entropy Principle (MEP). It automatically handles stoichiometry adjustments by introducing chalcogen vacancies and mapping multiple metal substitutions to the lattice. The script then optimizes the atomic configuration through an iterative swapping process (Monte Carlo) to balance the entropic force based on the theoretical free volume of the elements. Finally, it evaluates the space group symmetry and exports the optimized structures into CIF and POSCAR formats along with a detailed statistical text report.

## Installation
```bash
pip install -r requirements.txt
