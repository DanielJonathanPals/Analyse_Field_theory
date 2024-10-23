import Pkg
Pkg.activate(".")

using StaticArrays

function f(l::Array{Float64})
    r=1
    s = @SVector [0.0, 0.0]
    for x in 1:100
        for y in 1:100
            s = @SVector [l[x,y,1], l[x,y,2]]
            a = det(s)
            l[x,y,:] .= a * s
        end
    end
end

function det(s::SArray{Tuple{2},Float64,1,2})
    r=1
    a = @SMatrix [s[1]*3 -s[2]; (s[1] + s[2]) -s[2]]
    return a
end

l = rand(100, 100, 2)
f(l)
@time f(l)