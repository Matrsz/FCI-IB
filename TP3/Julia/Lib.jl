module Lib

using Statistics
using Distributions
using LinearAlgebra
using Base.Iterators

function noise(N0)
    d = Normal(0, N0/2)
    return rand(d) + im*rand(d)
end

function noise(N0, n) 
    return [noise(N0) for i in 1:n]
end

from_dB(x) = 10^(x/10)
to_dB(x) = 10*log10(x)


function tx(x, h, N0)
    return h*x + noise(N0)
end

function tx(x, h, N0, L) 
    x_r = repetition(x, L)
    x_i = interleave(x_r, L)
    return diagm(h)*x_i + noise(N0, length(h))
end

function s_stat(y, h)
    return h'*y/abs(h)
end

function s_stat(y, h, L)
    h_i = interleave(h, L)
    y_i = interleave(y, L)
    return [h_j'*y_j./norm(h_j) for (y_j, h_j) in zip(partition(y_i, L), partition(h_i, L))]
end

function decide(r)
    return real(r) > 0 ? 1 : -1 
end

function decide(r, L)
    return [real(mean(x)) > 0 ? 1 : -1 for x in partition(r, L)]
end

function rx(y, h)
    r = s_stat(y, h)
    return decide(r)
end

function rx(y, h, L)
    r = s_stat(y, h, L)
    return decide.(r)
end

function getBER(h0, N0, m_len)
    x = rand([-1,1], m_len)
    r = tx.(x, h0, N0)
    y = rx.(r, h0)
    return count(x .!= y)/m_len
end 

function getBER(h, N0, m_len, L)
    x = [rand([-1, 1], L) for i in 1:div(m_len, L)]
    r = [tx(xi, hi, N0, L) for (xi, hi) in zip(x, partition(h, L^2))]
    y = [rx(yi, hi, L) for (yi, hi) in zip(r, partition(h, L^2))]
    return sum(count(xi .!= yi) for (xi, yi) in zip(x, y))/m_len
end 

function BER_E(snr, h0, n_points, m_len)
    N0s = sqrt.(2 ./snr)
    bers = zeros(length(snr), n_points)
    for i in 1:n_points
        bers[:, i] = getBER.(rand(h0), N0s, m_len)
    end
    return [mean(c) for c in eachrow(bers)]
end

function repetition(x, L)
    return vcat([repeat([i], L) for i in x]...)
end

function interleave(x, L)
    A = reshape(x, (L, L))
    return vcat([eachrow(A)...]...)
end

function genh0(h0, m_len, L, v)
    t = 1:m_len*L^2
    N = size(h0)[1]
    x0 = (rand(1:N), rand(1:N))
    dir = rand([-1,1]).*rand([(1,0),(0,1)])
    x(t) = mod.(x0 .+ v*t.*dir, N).+1
    geth0(x) = h0[x...]
    return t, geth0.(x.(t))
end

function BER_E(snr, h0, n_points, m_len, L, v)
    N0s = sqrt.(2 ./snr)
    bers = zeros(length(N0s), n_points)
    for i in 1:n_points
        t, h = genh0(h0, m_len, L, v)
        bers[:, i] = [getBER(h, N0, m_len, L) for N0 in N0s]
    end
    return [mean(c) for c in eachrow(bers)]
end

function BER_T(snr)
    μ = sqrt(snr/(1+snr))
    return (1-μ)/2
end

function BER_T(snr, L)
    μ = sqrt(snr/(1+snr))
    S = sum(binomial(L-1+l, l)*((1+μ)/2)^l for l in 0:L-1)
    return ((1-μ)/2)^L*S
end

end