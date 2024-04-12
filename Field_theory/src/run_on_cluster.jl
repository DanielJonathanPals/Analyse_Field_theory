include("run_simulation.jl")

ToInt64(str::String) = Int64(parse(Float64, str))

# Run the simulation
run_simulation(ToInt64(ARGS[1]))