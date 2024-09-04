#!/bin/bash

################ SLURM HEADER ################

#SBATCH --job-name=Lattice_model_simulations
#SBATCH --mail-type=ALL

#SBATCH --output=src_simulation/slurm_out/slurm-%A-%a-%N.out
#SBATCH --error=src_simulation/slurm_err/slurm-%A-%a-%N.err

#SBATCH --mem=3G
#SBATCH --time=24:00:00

################ BASH SCRIPT #################

export JULIA_CPU_TARGET="generic;sandybridge;icelake-server;znver2;haswell;broadwell;znver1;skylake-avx512;znver3;cascadelake"
module load julia/1.10.3
julia --heap-size-hint=2G src_run_gillespie/run_on_cluster.jl $SLURM_ARRAY_TASK_ID
module unload julia
