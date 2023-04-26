

struct AttributeDefs
  func::Dict{Int,Function}
  name::Dict{Int,String}
  symb::Dict{Int,Symbol}
  number::Dict{Symbol,Int}
  
  AttributeDefs() = new(Dict{Int,Function}(),Dict{Int,String}(),Dict{Int,Symbol}(),Dict{Symbol,Int}())
end

function addAtt!(dic::AttributeDefs,s::Symbol,f::Function,n::String)
  i = length(dic.func)+1
  dic.symb[i] = s
  dic.func[i] = f
  dic.name[i] = n
  dic.number[s] = i
  ;
end


function attMap(dt::DataFrame,dic::AttributeDefs)
  L = size(dt)[1]
  D = length(dic.symb)
  as = zeros(Bool,L,D)
  for i in 1:L
    for j in 1:D
      as[i,j] = dic.func[j](dt[i,:])
    end
  end
  as
end


mutable struct Data
  ResR::Vector{Float64}
  ResI::Vector{Float64}
  PrR::Vector{Float64}
  PrI::Vector{Float64}
  site::Vector{String}
  freq::Vector{Float64}
  noise::Vector{Float64}
  as::Matrix{Bool}
  dic::AttributeDefs
  freqNr::Int
  dLogFreq::Float64
  pruneplus::Float64
  D::Int
  L::Int
  
  function Data(dt::DataFrame,dic::AttributeDefs,freqNr::Int)
    as = attMap(dt,dic)
    dlf = log10(dt[1,:freq])-log10(dt[2,:freq])
    new(dt[:,:ResR],dt[:,:ResI],dt[:,:PrR],dt[:,:PrI],dt[:,:site],dt[:,:freq],dt[:,:noise],as,dic,freqNr,dlf,0.0,size(as)[2],size(as)[1])
  end
end


Base.length(d::Data) = d.L


function getAtt(dat::Data,s::Symbol,z)
  i = get(dat.dic.number,s,nothing)
  if isnothing(i) ; error("attribute $s not recognised!") ; end
  d.as[z,i]
end


mutable struct Leaf
  p::Float64 # leaf parameters
  N::Int # number of rows mapped to this leaf
  v::Float64 # misfit at this leaf
end

Leaf() = Leaf(0.0,0,0.0)


mutable struct Split
  con::Int # split condition attribute
  t1::Union{Split,Leaf}
  t2::Union{Split,Leaf}
  N::Int
  v::Float64
end

Split(a::Int,b,c) = Split(a,b,c,0,0.0)

function Split(a::Symbol,b,c)
  i = findfirst(x->x==a,dic.symb)
  if isnothing(i) ; error("attribute $a not recognised!") ; end
  Split(i,b,c,0,0.0)
end


abstract type Variogram end


mutable struct ExpVariogram <: Variogram
  rho::Float64
  ex::Float64
end


mutable struct Tree
  root::Union{Split,Leaf}
  vg::Variogram
end


function saveTree(path::String,tr::Tree)
  if !endswith(path,".jld")
    error("tree must be saved in a file ending in \".jld\", invalid path: $path")
  end
  save(path,"root",tr.root,"vg",tr.vg)
  ;
end

function loadTree(path::String)
  if !endswith(path,".jld")
    error("tree input file must end in \".jld\", invalid path: $path")
  end
  d = load(path)
  Tree(d["root"],d["vg"])
end


function leaffor(n::Leaf,dat::Data,i::Int)
  n
end

function leaffor(n::Split,dat::Data,i::Int)
  dat.as[i,n.con] ? leaffor(n.t1,dat,i) : leaffor(n.t2,dat,i)
end

function leaffor(t::Tree,dat::Data,i::Int)
  leaffor(t.root,dat,i)
end


function splitUs(dat::Data,us::Vector{Bool},con::Int)
  L = length(us)
  @assert length(dat) == L
  us1 = zeros(Bool,L) ; us2 = zeros(Bool,L)
  for i in 1:L
    us1[i] = us[i] && dat.as[i,con]
    us2[i] = us[i] && !dat.as[i,con]
  end
  us1,us2
end







