using Base.Threads
using StringDistances
using LightXML
using BenchmarkTools
using Random
using Plots; gr()

function normalize_str(c)
    c = replace(c, "\n" => "")
    c = replace(c, "\t" => "")
    c = replace(c, "  " => "")
    c = strip(c)
    c = uppercase(c)
    c
end

function distance_mat(c, d)
    n = length(c)
    n = length(c)
    m = Matrix(undef, n, n)

    @threads for i in 1:size(m)[1]
        @inbounds m[i, :] .= d.(c[i], c)
    end

    m
end

function centers_distance!(p, c, d, m)
    for k in 1:length(c)
        @inbounds m[:, k] .= d.(p[c[k]], p)
    end
end

function row_minimum(m)
    mindist = Array{Int64}(undef, size(m, 1))
    @threads for i in 1:size(m, 1)
        @inbounds mindist[i] = findfirst(m[i,:] .== minimum(m[i,:]))
    end
    mindist
end

function kmeans(strings, distfun, k, convtol = 1e-2, maxiter=25)
    centers = randperm(length(strings))[1:k]
    centers_mat = Matrix{Int64}(undef, length(strings), length(centers))

    converged = false
    iters = 0
    prior = 0
    distances = []
    while !converged | iters < maxiter
        iters += 1
        centers_distance!(strings, centers, d, centers_mat)
        assignments = row_minimum(centers_mat)

        cdist = 0
        for k in 1:length(centers)
            c_p = [strings[i] for i in 1:length(strings) if assignments[i] == k]
            if length(c_p) > 0
                dm = distance_mat(c_p, d)
                md = mean(dm, dims=2)
                new_center = findfirst(md[:] .== minimum(md))
                centers[k] = new_center
                cdist += sum(md)
            end
        end

        if iters > 1
            converged = abs(cdist/prior - 1) <= convtol
        end

        prior = cdist
        append!(distances, [cdist])
    end

    return centers, row_minimum(centers_mat), distances
end

files = readdir("blogs")
xposts = []

for f in files
    x = parse_file(joinpath("blogs", files[1]))
    r = root(x)
    append!(xposts, get_elements_by_tagname(r, "post"))
end

d = Levenshtein()
posts = normalize_str.(content.(xposts))
res = kmeans(posts, d, 3, 1e-5, 10)
