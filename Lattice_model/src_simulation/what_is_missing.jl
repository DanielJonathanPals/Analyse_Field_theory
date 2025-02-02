file_name = "epsilon_-2p95_rho_v_0p05_dmu_-1p0_long_field"


ToInt64(str::String) = Int64(parse(Float64, str))

simulation_file_names = readdir("Data/" * file_name)
filter!(s->occursin(r"simulation_No", s),simulation_file_names)

numbers = [ToInt64(string(split(simulation_file_name, "_")[end])) for simulation_file_name in simulation_file_names]

diff = setdiff(1:256, numbers)
println(diff)