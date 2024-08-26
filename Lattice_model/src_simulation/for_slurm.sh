#!/bin/bash

################ SLURM HEADER ################

#SBATCH --job-name=Lattice_model_simulations
#SBATCH --mail-type=ALL

#SBATCH --output=src_simulation/slurm_out/slurm-%A-%a-%N.out
#SBATCH --error=src_simulation/slurm_err/slurm-%A-%a-%N.err

#SBATCH --mem=10G
#SBATCH --time=00:10:00

################ BASH SCRIPT #################

module load julia/1.10.3
julia --heap-size-hint=5G src_run_gillespie/run_on_cluster.jl $SLURM_ARRAY_TASK_ID
module unload julia
