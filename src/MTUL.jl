module MTUL

using
  Revise, DataFrames, CSV, Distributions, JLD, Optim, StaticArrays, LinearAlgebra, GraphViz, Plots

export
  attributeDefs1, tree1, 
  readTrainingData, readRealData,
  bestTree, expandTree, fitTree, 
  saveTree, loadTree, treePdf, saveTreeTxt,
  randLines, exportResidualData


include("tree.jl")
include("mod.jl")
include("inout.jl")


function readTrainingData(datafile::String,dic::AttributeDefs,freqNr::Int,useDet::Bool=false)
  @assert freqNr > 1
  dd = CSV.read(datafile,DataFrame,header=true)
  L = size(dd)[1]
  @assert rem(L,freqNr) == 0
  dat = DataFrame()
  if hasproperty(dd,:site)
    dat[!,:site] = string.(dd[:,:site])
  else
    sns = String[ "S"*string(div(i-1,freqNr)+1) for i in 1:L ]
    dat[!,:site] = sns
  end
  dat[!,:Zxyr] = log.(abs.(Float64.(dd[:,:Zxyr])))
  dat[!,:Zxyi] = log.(abs.(Float64.(dd[:,:Zxyi])))
  dat[!,:Zyxr] = log.(abs.(-Float64.(dd[:,:Zyxr])))
  dat[!,:Zyxi] = log.(abs.(-Float64.(dd[:,:Zyxi])))
  dat[!,:freq] = Float64.(dd[:,Symbol("freq(Hz)")])
  dat[!,:beta] = Float64.(dd[:,:beta])
  dat[!,:elip] = Float64.(dd[:,:ellipticity])
  dat[!,:betaM] = zeros(L)
  dat[!,:elipM] = zeros(L)
  for i in 1:L
    j = mod1(i,freqNr)
    if j == 1
      dat[i,:betaM] = abs(dat[i,:beta])
      dat[i,:elipM] = abs(dat[i,:elip])
    else
      dat[i,:betaM] = max(dat[i-1,:betaM],abs(dat[i,:beta]))
      dat[i,:elipM] = max(dat[i-1,:elipM],abs(dat[i,:elip]))
    end
  end
  if useDet
    mv = Matrix{Complex{Float64}}[ ([(dd[i,:Zxxr]+dd[i,:Zxxi]im) (dd[i,:Zxyr]+dd[i,:Zxyi]im) ; (dd[i,:Zyxr]+dd[i,:Zyxi]im) (dd[i,:Zyyr]+dd[i,:Zyyi]im)]) for i in 1:L ]
    dets = sqrt.(det.(mv))
    dat[!,:PrR] = log.(abs.(real.(dets)))
    dat[!,:PrI] = log.(abs.(imag.(dets)))
  else
    dat[!,:PrR] = (dat[:Zxyr] .+ dat[:Zyxr]) ./ 2
    dat[!,:PrI] = (dat[:Zxyi] .+ dat[:Zyxi]) ./ 2
  end
  dat[!,:difPol] = abs.(dat[:,:Zxyr] - dat[:,:Zyxr]) + abs.(dat[:,:Zxyi] - dat[:,:Zyxi])
  dat[!,:Zr] = log.(Float64.(dd[:,:Zr]))
  dat[!,:Zi] = log.(Float64.(dd[:,:Zi]))
  dat[!,:ResR] = dat[:,:Zr] .- dat[:,:PrR]
  dat[!,:ResI] = dat[:,:Zi] .- dat[:,:PrI]
  # dat[:noise] does not affect training
  dat[!,:noise] = hasproperty(dd,:noise) ? Float64.(dd[:,:noise]) : zeros(L)
  Data(dat,dic,freqNr)
end


function readRealData(datafile::String,dic::AttributeDefs,useDet::Bool=false)
  dd = CSV.read(datafile,header=true)
  L = size(dd)[1]
  dat = DataFrame()
  dat[!,:site] = string.(dd[:,:site])
  dat[!,:Zxyr] = log.(abs.(Float64.(dd[:,:Zxyr])))
  dat[!,:Zxyi] = log.(abs.(Float64.(dd[:,:Zxyi])))
  dat[!,:Zyxr] = log.(abs.(-Float64.(dd[:,:Zyxr])))
  dat[!,:Zyxi] = log.(abs.(-Float64.(dd[:,:Zyxi])))
  dat[!,:freq] = Float64.(dd[:,Symbol("freq(Hz)")])
  dat[!,:beta] = Float64.(dd[:,:beta])
  dat[!,:elip] = Float64.(dd[:,:ellipticity])
  dat[!,:betaM] = zeros(L)
  dat[!,:elipM] = zeros(L)
  st1 = "nothing"
  for i in 1:L
    st2 = dat[i,:site]
    if st1 != st2
      dat[i,:betaM] = abs(dat[i,:beta])
      dat[i,:elipM] = abs(dat[i,:elip])
    else
      dat[i,:betaM] = max(dat[i-1,:betaM],abs(dat[i,:beta]))
      dat[i,:elipM] = max(dat[i-1,:elipM],abs(dat[i,:elip]))
    end
    st1 = st2
  end
  if useDet
    mv = Matrix{Complex{Float64}}[ ([(dd[i,:Zxxr]+dd[i,:Zxxi]im) (dd[i,:Zxyr]+dd[i,:Zxyi]im) ; (dd[i,:Zyxr]+dd[i,:Zyxi]im) (dd[i,:Zyyr]+dd[i,:Zyyi]im)]) for i in 1:L ]
    dets = sqrt.(det.(mv))
    dat[!,:PrR] = log.(abs.(real.(dets)))
    dat[!,:PrI] = log.(abs.(imag.(dets)))
  else
    dat[!,:PrR] = (dat[:,:Zxyr] .+ dat[:,:Zyxr]) ./ 2
    dat[!,:PrI] = (dat[:,:Zxyi] .+ dat[:,:Zyxi]) ./ 2
  end
  dat[!,:difPol] = abs.(dat[:,:Zxyr] - dat[:,:Zyxr]) + abs.(dat[:,:Zxyi] - dat[:,:Zyxi])
  dat[!,:Zr] = deepcopy(dat[:,:PrR])
  dat[!,:Zi] = deepcopy(dat[:,:PrI])
  dat[!,:ResR] = dat[:,:Zr] .- dat[:,:PrR]
  dat[!,:ResI] = dat[:,:Zi] .- dat[:,:PrI]
  if hasproperty(dd,:stdXY)
    cent = sqrt.((exp.(dat[:,:PrR]) .^ 2) .+ (exp.(dat[:,:PrI]) .^ 2))
    h(x) = Float64(x)^2
    sd = h.(dd[:,:stdXY]) .+ h.(dd[:,:stdYX]) .+ h.(dd[:,:stdXX]) .+ h.(dd[:,:stdYY])
    sd = sqrt.(sd)
    dv1 = log.(1 .+ (sd ./ cent))
    dv2 = deepcopy(dv1)
    for i in 1:L
      @assert sd[i] >= 0 "non-positive processing noise encountered!"
      if sd[i] < cent[i]*0.9
        dv2[i] = -log.(1 - sd[i]/cent[i])
      else
        println("\nWarning: processing noise > 90% encountered at i == $(i)!\n")
      end
    end
    dat[!,:noise] = (dv1 .+ dv2) ./ (2*sqrt(2))
  else
    dat[!,:noise] = hasproperty(dd,:noise) ? Float64.(dd[:,:noise]) : zeros(L)
  end
  Data(dat,dic,0)
end


function attributeDefs1()
  dic = AttributeDefs()

  addAtt!(dic,:elip1,x -> x[:elipM] > 0.1,"elip>0.1")
  addAtt!(dic,:elip2,x -> x[:elipM] > 0.2,"elip>0.2")
  addAtt!(dic,:elip3,x -> x[:elipM] > 0.3,"elip>0.3")
  addAtt!(dic,:elip4,x -> x[:elipM] > 0.4,"elip>0.4")
  addAtt!(dic,:elip5,x -> x[:elipM] > 0.5,"elip>0.5")
  addAtt!(dic,:elip7,x -> x[:elipM] > 0.75,"elip>0.75")
  #addAtt!(dic,:elip10,x -> x[:elipM] > 1.0,"elip>1.0")
  addAtt!(dic,:beta1,x -> x[:betaM] > 1,"|beta|>1")
  addAtt!(dic,:beta2,x -> x[:betaM] > 2,"|beta|>2")
  addAtt!(dic,:beta3,x -> x[:betaM] > 3,"|beta|>3")
  addAtt!(dic,:beta5,x -> x[:betaM] > 5,"|beta|>5")
  addAtt!(dic,:beta7,x -> x[:betaM] > 7,"|beta|>7")
  addAtt!(dic,:beta9,x -> x[:betaM] > 9,"|beta|>9")
  #addAtt!(dic,:beta12,x -> x[:betaM] > 12,"|beta|>12")
  #addAtt!(dic,:difPol1,x -> x[:difPol] > 0.25,"difPol>0.25")
  #addAtt!(dic,:difPol0,x -> x[:difPol] > 0.05,"difPol>0.05")
  addAtt!(dic,:difPol1,x -> x[:difPol] > 0.1,"difPol>0.1")
  #addAtt!(dic,:difPol2,x -> x[:difPol] > 0.2,"difPol>0.2")
  addAtt!(dic,:difPol3,x -> x[:difPol] > 0.3,"difPol>0.3")
  #addAtt!(dic,:difPol4,x -> x[:difPol] > 0.4,"difPol>0.4")
  addAtt!(dic,:difPol5,x -> x[:difPol] > 0.5,"difPol>0.5")
  #addAtt!(dic,:difPol6,x -> x[:difPol] > 0.6,"difPol>0.6")
  #addAtt!(dic,:difPol7,x -> x[:difPol] > 0.7,"difPol>0.7")
  #addAtt!(dic,:difPol8,x -> x[:difPol] > 0.8,"difPol>0.8")
  #addAtt!(dic,:difPol3,x -> x[:difPol] > 0.5,"difPol>0.5")
  dic
end


function tree1(dic::AttributeDefs)
  a = Split(dic.number[:beta3],Leaf(),Leaf())
  c = Split(dic.number[:elip1],a,Leaf())
  Tree(c,ExpVariogram(4.0,1.5))
end


function tree0(dic::AttributeDefs)
  Tree(Leaf(),ExpVariogram(4.0,1.5))
end


end
