struct Field
    state::Array{Float64,2}
end

function f()
    x_size = 128
    y_size = 192
    numb_snapshots = 100
    B = Matrix{Float64}(undef, (x_size * numb_snapshots, y_size))
    f = Field(zeros(Float64, x_size, y_size))
    for i in 1:numb_snapshots
        f.state[:,:] = rand(x_size, y_size)
        B[((i-1) * x_size + 1):(i * x_size), :] = f.state
    end
    return B
end

f()