#!/bin/bash

# Navigate to the Lattice_model directory and then run "bash src_simulation/run_simulation.sh" in the terminal
# The parameters can be changed in the file "simulation_parameters.txt"


# Determine number of array jobs
file="src_simulation/simulation_parameters.txt"
total_simulations=0
line_number=0
while IFS=',' read -r epsilon rho_v dmu t_max save_interval numb_of_simulations f_res_init_guess || [ -n "$numb_of_simulations" ]
do
    line_number=$((line_number + 1))
    if [ "$line_number" -le 2 ]; then
        continue
    fi
    numb_of_simulations=$(echo $numb_of_simulations | xargs)
    total_simulations=$((total_simulations + numb_of_simulations))
done < "$file"

# set up file structure
module load python/3.7.4
/bin/python3 /scratch/d/Daniel.Pals/Masterthesis/Coding/Analyse_Field_theory/Lattice_model/src_parameter_processing/process_params.py
module unload python

module load julia/1.11.0
julia src_parameter_processing/Save_parameters.jl
module unload julia

# Run the simulations
sbatch --array=1-$total_simulations src_simulation/for_slurm.sh