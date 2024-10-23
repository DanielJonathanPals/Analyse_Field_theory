folder = "epsilon_-2p5_rho_v_0p05_dmu_1p0"

ToInt64(str::String) = Int64(parse(Float64, str))

simulation_file_names = readdir("Data/" * folder)
filter!(s->occursin(r"simulation_No", s),simulation_file_names)

numbers = [ToInt64(string(split(simulation_file_name, "_")[end])) for simulation_file_name in simulation_file_names]

what_is_missing = setdiff(1:256, numbers)

large_numbers = numbers
filter!(n -> n > 256, large_numbers)

for (i,missing_number) in enumerate(what_is_missing)
    if i > length(large_numbers)
        break
    end
    mv("Data/$folder/simulation_No_$(large_numbers[i])", "Data/$folder/simulation_No_$missing_number")
end
