module Lib

export genClust, genScatt, genDists, genhl, delaySpread

using FFTW
using Trapz
using Distributions
using Statistics

""" 
**clust = genClust(N; minR, minDist)** 

Genera un cluster de N posiciones uniformemente distribuidas en un
círculo de diámetro unitario. Retorna un vector de tuplas (x,y).

El parámetro opcional minR indica una distancia mínima al centro del cluster que deben tener los elementos.

El parámetro opcional minDist indica una distancia mínima a otros elementos que deben tener los elementos.
"""
function genClust(N, minR=0, minDist=0)
    clust = []
    for i in 1:N
        xy = (0, 0)
        j = 0
        while true
            j = j+1
            r = sqrt(rand())
            t = 2π*rand()
            xy = (r*cos(t), r*sin(t))
            hypot(xy...)<minR || any(p -> hypot((xy.-p)...) < minDist*0.9^j, clust) || break
        end
        push!(clust, xy)
    end
    return clust
end

"""
**S, Ss = genScatt(Ns, Rs, Cs)**

Genera una distribución de reflectores organizados en clusters
definidos por los vectores Ns, Rs, Cs de la siguiente forma

Ns[i] número de reflectores en el cluster i-ésimo

Rs[i] radio del cluster i-ésimo

Cs[i] coordenadas del centro del cluster i-ésimo

Retorna S un vector de tuplas (x, y) representando las posiciones
de los reflectores, y un vector de vectores Ss representando las
posiciones de los reflectores agrupados en sus respectivos clusters.
"""
function genScatt(Ns, Rs, Cs)
    clust(N, R, C) = [C .+ R .* x for x in genClust(N)]
    cs = [clust(N, R, C) for (R, C, N) in zip(Rs, Cs, Ns)] 
    return [(cs... )... ], cs
end

"""
**dists = genDists(S, posTx, posRx)**

Recibe un vector de tuplas S representanto las posiciones (x, y)
de reflectores, y dos tuplas posTx y posRx representando las posiciones
de un emisor y receptor.

Retorna un vector con las distancias de todos los posibles caminos entre 
el emisor y el receptor.
"""
function genDists(S, posTx, posRx, return_TxRx=false)
    Tx_dists = [hypot((posTx.-s)...) for s in S]
    Rx_dists = [hypot((posRx.-s)...) for s in S]
    if return_TxRx
        return Tx_dists, Rx_dists
    end
    return Tx_dists.+Rx_dists
end


function genAtts(dists)
    return [-1/x for x in dists]
end


""" 
**hl, tt = genhl(Dsts, fc, W, a_i, oversample, margin)**

Esta funcion produce la respuesta impulsiva discreta equivalente banda base 
del canal correspondiente a todos los caminos con distancias dadas por el 
vector Dsts con atenuaciones a_i, a una frecuencia central fc y un ancho de banda W.
Recibe los siguientes parámetros opcionales:

oversample: Un parametro que debe ser un número entero mayor o igual a 1 y 
que sirve para generar puntos intermedios entre las muestras (para simular 
el comportamiento "analogico" del canal)

margin: la respuesta se calcula para floor(margin)/2 muestras anteriores al
primer eco (o camino) recibido del canal e idem para el ultimo recibido. 
"""
function genhl(dists, fc, W, a_i, oversample=1, margin=0)
    delays = dists./3e8
    groupdelay = minimum(delays)
    Td = maximum(delays)-groupdelay
    rel_delays = delays .- groupdelay
    a_ib = a_i .* exp.(-im*2π*fc.*delays)
    len = floor(Td*W*oversample)+margin*oversample

    hl = [sinc.(l/oversample .- floor(margin/2) .- rel_delays.*W)'*a_ib for l in 0:len]
    tt = (0:len)./W./oversample .- floor(margin/2)/W .+ groupdelay

    return hl, tt
end

""" 
**Ds = delaySpread(Dsts)**

Recibe como parametro un vector dists de distancias (en metros)
y retorna el delay spread Td (en segundos) entre esas distancias.
""" 
function delaySpread(dists)
    return (maximum(dists)-minimum(dists))/3e8
end

function genHl(hl, tt)
    power(x, y) = trapz(x, abs.(y).^2)
    ff = fftfreq(length(tt), 1/(tt[2]-tt[1])) |> fftshift
    Hl = fft(hl) |> fftshift
    Hl = Hl.*sqrt(power(tt, hl)/power(ff, Hl))
    return Hl, ff
end

function simulateFlatFade(nRealz; Ns, Rs, Cs, posTx, posRx, fc, W, a_i, margin)
    hs = []
    for i in 1:nRealz
        S, _ = Lib.genScatt(Ns,Rs,Cs)
        dsts = Lib.genDists(S,posTx,posRx)
        h, _ = Lib.genhl(dsts,fc,W,a_i,1,margin)
        imax = argmax(abs.(h))
        push!(hs, h[imax])
    end
    return hs./mean(abs.(hs))
end

function toRician(a_i, k)
    oldpower  = sum(abs.(a_i).^2)
    a_i[1] = sqrt(k*sum(abs.(a_i[2:end]).^2))
    newpower = sum(abs.(a_i).^2)
    return a_i.*sqrt(oldpower/newpower)
end

function simulateFlatFade(nRealz, k; Ns, Rs, Cs, posTx, posRx, fc, W, a_i, margin)
    a_i = toRician(a_i, k)
    hs = simulateFlatFade(nRealz; Ns, Rs, Cs, posTx, posRx, fc, W, a_i, margin)
    abs_hs = abs.(hs)
    omega = mean(abs_hs.^2)
    sigma = sqrt(omega/(2*(k+1)))
    nu = sigma*sqrt(2*k)
    return hs, Rician(nu, sigma)  
end

function simulateDoppler(posTx, x, S, fc, W, margin)
    function geth0(xi)
        Txd, Rxd = Lib.genDists(S,posTx,xi, true)
        a_i = 1 ./(Txd.*Rxd)
        h, _ = Lib.genhl(Txd.+Rxd,fc,W,a_i,1,margin)
        imax = argmax(abs.(h))
        return h[imax]
    end
    return geth0.(x)
end

function simulateClarke(t, x, nRealz; N, R, posTx, posRx, fc, W, margin)
    function autocor(x)
        M = x*x'
        imid = Int(length(x)/2)
        return [M[i, imid] for i in eachindex(x)]
    end
    S, _ = Lib.genScatt([N], [R], [posRx])
    h = Lib.simulateDoppler(posTx, x.(t), S, fc, W, margin)
    rs = zeros(Complex{Float64}, length(h), nRealz)
    for i in 1:nRealz
      S, _ = Lib.genScatt([N], [R], [posRx])
      h = Lib.simulateDoppler(posTx, x.(t), S, fc, W, margin)
      rs[:,i] = autocor(h)
    end
    r = [mean(rs[i,:]) for i in eachindex(t)] 
    r = r ./ maximum(abs.(r))
    tt = t.-t[end]/2
    return tt, r
end
end

