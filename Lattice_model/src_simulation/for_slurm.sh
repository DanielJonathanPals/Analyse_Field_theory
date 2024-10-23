#!/bin/bash

################ SLURM HEADER ################

#SBATCH --job-name=Lattice_model_simulations
#SBATCH --mail-type=ALL

#SBATCH --output=src_simulation/slurm_out/slurm-%A-%a-%N.out
#SBATCH --error=src_simulation/slurm_err/slurm-%A-%a-%N.err

#SBATCH --mem=3G
#SBATCH --time=24:00:00

#SBATCH --exclude=th-cl-rome01n1,th-cl-rome01n2,th-cl-rome01n3,th-cl-rome01n4,th-cl-rome02n1,th-cl-rome02n2,th-cl-rome02n3,th-cl-rome02n4,th-cl-rome03n1,th-cl-rome03n2,th-cl-rome03n3,th-cl-rome03n4,th-cl-rome04n1,th-cl-rome04n2,th-cl-rome04n3,th-cl-rome04n4,th-cl-rome05n1,th-cl-rome05n2,th-cl-rome05n3,th-cl-rome05n4,th-cl-rome06n1,th-cl-rome06n2,th-cl-rome06n3,th-cl-rome06n4,th-cl-rome07n1,th-cl-rome07n2,th-cl-rome07n3,th-cl-rome07n4,th-cl-rome08n1,th-cl-rome08n2,th-cl-rome08n3,th-cl-rome08n4,th-cl-rome09n1,th-cl-rome09n2,th-cl-rome09n3,th-cl-rome09n4,met-cl-lx017,met-cl-lx018,met-cl-lx019,met-cl-lx020,met-cl-lx022,met-cl-lx023,met-cl-lx024,met-cl-lx025,met-cl-vis01,met-cl-vis02,th-cl-1024us06,th-cl-1024us04,th-cl-1024us05,usm-cl-826bac01,usm-cl-826bac03 

################ BASH SCRIPT #################

cat <<- EOF
        ${HOSTNAME}
        job id  ${SLURM_JOB_ID}
        job start at `date`
        ${WORKDIR}
EOF
########################


export JULIA_CPU_TARGET="generic;sandybridge;icelake-server;znver2;haswell;broadwell;znver1;skylake-avx512;znver3;cascadelake"
module load julia/1.11.0
julia --heap-size-hint=2G src_run_gillespie/run_on_cluster.jl $SLURM_ARRAY_TASK_ID
module unload julia
