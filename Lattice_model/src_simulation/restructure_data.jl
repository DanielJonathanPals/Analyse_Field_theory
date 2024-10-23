name = "epsilon_-2p5_rho_v_0p05_dmu_-0p2"

target_folder = name
move_folder = "$(name)_old"

ToInt64(str::String) = Int64(parse(Float64, str))

simulation_file_names = readdir("Data/" * target_folder)
filter!(s->occursin(r"simulation_No", s),simulation_file_names)

numbers = [ToInt64(string(split(simulation_file_name, "_")[end])) for simulation_file_name in simulation_file_names]

what_is_missing = setdiff(1:256, numbers)

move_file_names = readdir("Data/" * move_folder)
filter!(s->occursin(r"simulation_No", s),move_file_names)


for (i,missing_number) in enumerate(what_is_missing)
    if i > length(move_file_names)
        break
    end
    mv("Data/$move_folder/$(move_file_names[i])", "Data/$target_folder/simulation_No_$missing_number")
end
