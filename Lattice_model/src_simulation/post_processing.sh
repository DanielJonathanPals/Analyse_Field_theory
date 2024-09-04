# run this after the slurm job has finished by running "bash src_simulation/post_processing.sh" in the terminal

module load julia/1.10.3
julia src_analysis/compute_expectation_hq2.jl
module unload julia