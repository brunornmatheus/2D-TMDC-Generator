#!/usr/bin/env python

# mcmaxent_2d.py

# ASE (Atomic Simulation Environment) python script to
# produce supercells of 2D TMDC materials (MX2) with vacancies and 
# metal substitutions according to the maximum entropy principle.

import sys
import numpy as np
import random as rnd
import argparse
import re
from datetime import datetime
from ase.formula import Formula
from ase import io
from ase.build import mx2, make_supercell
from ase.build.tools import sort
from ase.data import covalent_radii
import spglib

# ================= HELPER FUNCTIONS =================

def dists(indice, supercelula, Lista):
    """
    Calculate distances from atom 'indice' to all other atoms of the SAME element.
    mic=True accounts for periodic boundaries.
    """
    symbol = supercelula[indice].symbol
    # Safety check if element exists in list
    if symbol in Lista and len(Lista[symbol]) > 0:
        dists = supercelula.get_distances(indice, Lista[symbol], mic=True)
        return dists
    return [0.0]

def dists_all(indice, supercelula):
    """
    Calculate distances from atom 'indice' to ALL atoms in the supercell,
    accounting for periodic boundary conditions (mic=True).
    """
    all_indices = list(range(len(supercelula)))
    distances = supercelula.get_distances(indice, all_indices, mic=True)
    return np.sort(distances)

def freeV(supercelula, use_covalent_radii=True):
    """
    Calculates the physical Free Volume: V_free = V_cell - V_occupied.
    
    Parameters:
    - supercelula: ASE Atoms object.
    - use_covalent_radii: If True, uses standard ASE covalent radii for each atom.
                          If False, uses nearest-neighbor distance across ALL species (d_NN / 2).
    """
    total_cell_vol = supercelula.get_volume()
    occupied_vol = 0.0

    if use_covalent_radii:
        for atom in supercelula:
            # Look up covalent radius by atomic number
            r = covalent_radii[atom.number]
            occupied_vol += (4.0 / 3.0) * np.pi * (r ** 3)
    else:
        for i in range(len(supercelula)):
            d = dists_all(i, supercelula)
            if len(d) > 1:
                r = d[1] / 2.0  # d[0] is self (0.0), d[1] is nearest neighbor
                occupied_vol += (4.0 / 3.0) * np.pi * (r ** 3)

    return total_cell_vol - occupied_vol

def get_element_statistics(supercelula, Lista, tavV):
    """
    Updated statistics output using nearest-neighbor distances across ALL species.
    """
    statistics = []
    statistics.append('Element\t| Theor. Free Vol.\t| Mean Occup. Vol.\t| Std. Deviation')
    
    for el in Lista:
        el_vols = []
        for idx in Lista[el]:
            d = dists_all(idx, supercelula)
            if len(d) > 1:
                r = d[1] / 2.0
                el_vols.append((4.0 / 3.0) * np.pi * (r ** 3))
            else:
                r = covalent_radii[supercelula[idx].number]
                el_vols.append((4.0 / 3.0) * np.pi * (r ** 3))
        
        mean_val = np.mean(el_vols) if el_vols else 0.0
        std_val = np.std(el_vols) if el_vols else 0.0
        statistics.append(f"{el}\t| {tavV[el]:8.4f}\t| {mean_val:8.4f}\t| {std_val:8.4f}")
        
    return statistics

def el_entropia_for(elemento, supercelula, Lista, tavV):
    """
    Calculate "Entropic Force" for a specific element.
    Summation of volumetric differences regarding the theoretical max volume.
    """
    el_entropia_for = 0
    if elemento not in Lista: return 0
    
    for i in Lista[elemento]:
        orddists = sorted(dists(i, supercelula, Lista))
        if len(orddists) > 1:
            v = orddists[1]**3 
            el_entropia_for += abs(v - tavV[elemento])
    return el_entropia_for

def troca(indice1, indice2, supercelula, Lista):
    """
    Swap two atoms in the supercell and update the index map.
    """
    sym1 = supercelula[indice1].symbol
    sym2 = supercelula[indice2].symbol
    
    supercelula[indice1].symbol = sym2
    supercelula[indice2].symbol = sym1
    
    # Update dictionary indices
    Lista[sym1].remove(indice1)
    Lista[sym1].append(indice2)
    Lista[sym2].remove(indice2)
    Lista[sym2].append(indice1)

def parse_arguments():
    parser = argparse.ArgumentParser(description='2D TMDC Generator with Vacancies (qbfmaxent style)')
    parser.add_argument('--f', type=str, default='MoWXYZ', help='Target Formula in Hill notation')
    parser.add_argument('--x', type=int, default=6, help='Supercell repetitions in X')
    parser.add_argument('--y', type=int, default=6, help='Supercell repetitions in Y')
    parser.add_argument('--z', type=int, default=1, help='Supercell repetitions in Z')
    parser.add_argument('--vacuum', type=float, default=15.0, help='Vacuum padding in Angstroms')
    parser.add_argument('--base_metal', type=str, default='Mo', help='Base Metal symbol (e.g., Mo)')
    parser.add_argument('--base_chalc', type=str, default='S', help='Base Chalcogen symbol (e.g., S)')
    parser.add_argument('--verbose', nargs='?', default='n', help='Enable verbose output (y/n)')
    parser.add_argument('--n1', type=int, default=1, help='Start index for file generation')
    parser.add_argument('--n2', type=int, default=1, help='End index for file generation')
    return parser.parse_args()

# ================= MAIN LOGIC =================

print('\b')
t_inicio = datetime.now()

args = parse_arguments()
verbose = (args.verbose != 'n')

# 1. Parse Target Composition
target_counts = Formula(args.f).count()
total_target_atoms = sum(target_counts.values())
chalcogen_symbol = args.base_chalc

# 2. Build Base Lattice (Full MX2)
print('-- qbfmaxent 2D v2 --')
print(f"Building base 2D lattice ({args.base_metal}{args.base_chalc}2)...")
base_layer = mx2(formula=f"{args.base_metal}{args.base_chalc}2", kind='2H', a=3.18, thickness=3.19)
P = [[args.x, 0, 0], [0, args.y, 0], [0, 0, 1]]
supercelula = make_supercell(base_layer, P)
supercelula.center(axis=2, vacuum=args.vacuum / 2)

# 3. Vacancy Handling (Adjust Atom Counts)
current_chalc_indices = [i for i, a in enumerate(supercelula) if a.symbol == chalcogen_symbol]
required_chalc = int(target_counts.get(chalcogen_symbol, 0))

if required_chalc < len(current_chalc_indices):
    to_remove = len(current_chalc_indices) - required_chalc
    print(f" Stoichiometry Adjustment: Removing {to_remove} {chalcogen_symbol} atoms to match formula.")
    
    # Pick random indices to delete
    indices_to_delete = rnd.sample(current_chalc_indices, to_remove)
    
    # Delete atoms (must sort indices reverse to not mess up order during deletion)
    del supercelula[[i for i in indices_to_delete]]
    
elif required_chalc > len(current_chalc_indices):
    print("Error: Target requires more Chalcogens than the lattice provides.")
    sys.exit(1)

# 4. Metal Substitution Preparation
metal_indices = [i for i, atom in enumerate(supercelula) if atom.symbol == args.base_metal]
required_metals = []
for el, count in target_counts.items():
    if el != chalcogen_symbol:
        required_metals.extend([el] * int(count))

if len(required_metals) != len(metal_indices):
    print(f"Error: Metal sites ({len(metal_indices)}) != Formula metals ({len(required_metals)}). Check Supercell Size.")
    sys.exit(1)

# Assign metals randomly initially
rnd.shuffle(required_metals)
for i, idx in enumerate(metal_indices):
    supercelula[idx].symbol = required_metals[i]

# 5. Build Index Map (Lista)
Lista = {}
for i, atom in enumerate(supercelula):
    if atom.symbol not in Lista: Lista[atom.symbol] = []
    Lista[atom.symbol].append(i)

# Theoretical Volume
tavV = dict((i, supercelula.cell.volume/len(Lista[i])*0.74) for i in Lista)

# Initial conditions display
print('\nNumber of atoms: '+str(len(supercelula)))
print('Supercell Chemical Formula: ', end='')
print(args.f)
print('\n')
print(f"Initial Free Volume: {freeV(supercelula):.4f}")
print('\b')

# Save Initial Structure
fname_base = f"{args.f}_2D_{args.x}x{args.y}"
io.write(f"{fname_base}_initial.cif", supercelula)
supercelula_sorted = sort(supercelula)
io.write('POSCAR_initial', supercelula_sorted, 'vasp')

# ================= OPTIMIZATION LOOP =================

for run_id in range(args.n1, args.n2+1):
    
    print(f"\n--- Run ID: {run_id} ---")

    # A. Shuffle Metals
    n_shuffle = len(metal_indices) * 5
    for k in range(n_shuffle):
        idx1 = rnd.choice(metal_indices)
        idx2 = rnd.choice(metal_indices)
        if supercelula[idx1].symbol != supercelula[idx2].symbol:
            troca(idx1, idx2, supercelula, Lista)
            if verbose and k % 100 == 0:
                 print(f'\r Shuffle step: {k}/{n_shuffle}', end='')
    
    if verbose: print('\nShuffle complete.')

    # Save Shuffle State
    io.write(f"{fname_base}_shuffle.cif", supercelula)

    # B. Optimize Entropic Force (Only Metals)
    elements = [e for e in Lista.keys() if e != chalcogen_symbol]
    improving = True
    scan_count = 0
    
    while improving:
        improving = False
        scan_count += 1
        
        for i in range(len(elements)):
            el1 = elements[i]
            for j in range(i+1, len(elements)):
                el2 = elements[j]
                
                pos1 = Lista[el1]
                pos2 = Lista[el2]
                
                for k, p1 in enumerate(pos1):
                    for h, p2 in enumerate(pos2):
                        
                        old_E = el_entropia_for(el1, supercelula, Lista, tavV) + \
                                el_entropia_for(el2, supercelula, Lista, tavV)
                        
                        troca(p1, p2, supercelula, Lista)
                        
                        new_E = el_entropia_for(el1, supercelula, Lista, tavV) + \
                                el_entropia_for(el2, supercelula, Lista, tavV)
                        
                        # Revert if energy (entropic force) increases
                        if new_E >= old_E:
                            troca(p1, p2, supercelula, Lista)
                        else:
                            improving = True
                            if verbose:
                                print(f'\rScan: {scan_count} Swap: {el1}-{el2} ({p1}-{p2})', end='')

    # ================= FINAL ANALYSIS =================
    
    final_fv = freeV(supercelula)
    print(f'\nFinal Free Volume: {final_fv:.4f}')

    # Statistics
    statistics = get_element_statistics(supercelula, Lista, tavV)

    print('\nStatistics:')
    for line in statistics:
        print(line)

    # Symmetry Analysis
    try:
        cell = (supercelula.get_cell(), supercelula.get_scaled_positions(), supercelula.get_atomic_numbers())
        grupo = spglib.get_spacegroup(cell, symprec=1e-5)
    except Exception:
        grupo = "Unknown/Error"
    print(f'Space Group: {grupo}')
    
    # Save Final Files
    hm = re.sub(r'/', '!', re.match(r'^\S+', grupo).group(0)) if grupo else "N_A"
    
    # CIF
    io.write(f"{fname_base}_run_{run_id:05d}_{hm}_final.cif", supercelula)
    
    # POSCAR
    supercelula_sorted = sort(supercelula)
    io.write(f"POSCAR_{run_id}", supercelula_sorted, 'vasp')

    # TXT Report
    report_name = f"{fname_base}.txt"
    with open(report_name, "a") as out:
        out.write('-- qbfmaxent 2D v2 --\n')
        out.write(f"{args.f} {args.x}x{args.y} {datetime.now()}\n")
        out.write(f"Run ID: {run_id}\n")
        out.write(f"Space Group: {grupo}\n")
        out.write(f"Final Free Volume: {final_fv:.4f}\n\n")
        for line in statistics:
            out.write(line + '\n')
        out.write('\nAtomic positions (scaled)\n')
        for atom in supercelula:
            out.write(f"{atom.symbol}\t{atom.scaled_position[0]:.6f}\t{atom.scaled_position[1]:.6f}\t{atom.scaled_position[2]:.6f}\n")
        out.write('\n--------------------------------------------------\n\n')

    # Log to summary file
    with open("2D_generation_log.txt", "a") as log:
        log.write(f"Run {run_id}: Formula={args.f} FV={final_fv:.4f} SG={grupo}\n")

print(f'\nExec. time: {datetime.now()-t_inicio}')
