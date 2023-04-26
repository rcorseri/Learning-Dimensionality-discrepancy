

function mix_nlogpdf(x,s)
  v = pdf(Normal(0,s),x)*0.95 + pdf(Cauchy(0,s),x)*0.05
  -log(v)
end


function trainLeaf!(dat::Data,us::Vector{Bool},lf::Leaf)
  N = count(us)
  lf.N = N
  if N <= 10
    lf.p = 0.0
    lf.v = Inf
    return Inf
  end
  rs = zeros(N*2)
  j = 1
  for i in 1:dat.L
    if !us[i] ; continue ; end
    rs[j] = dat.ResR[i]
    rs[j+1] = dat.ResI[i]
    j += 2
  end
  sd = median(abs.(rs))*1.483 # robust estimator of standard deviation
  lf.p = sd
  v = sum( mix_nlogpdf(r,sd) for r in rs )
  c = 0.0
  c += -log(20) + 10*lf.p
  c += -2.3
  c += -2*log(lf.p) + log(dat.L) + 0.5*log(4/3)
  lf.v = v + c + dat.pruneplus*log(dat.L)
  ;
end


spr() = 0.3 # prior probability of leaf split


function bestLeaf(dat::Data,us::Vector{Bool})
  lf = Leaf()
  trainLeaf!(dat,us,lf)
  lf
end


function maternCovar(d,rho)
  p = 1
  x = d*sqrt(2*p+1)/rho
  v = x
  v *= factorial(p)/factorial(2*p)
  z = 0.0
  for i in 0:p
    zz = factorial(p+i)/(factorial(i)*factorial(p-i))
    zz *= x^(p-i)
    z += zz
  end
  v *= z
  exp(-v)
end


function covar(d,vg::ExpVariogram)
  v = d/vg.rho
  v = v^vg.ex
  exp(-v)
end


# uses median estimator
# 2*gamma(h) = 2.198*median(empvg[h])^2
function varioLoss(empvg::Vector{Vector{Float64}},dLogFreq::Float64,ps)
  v = 0.0
  for i in 2:length(empvg)
    h = (i-1)*dLogFreq
    vg = ExpVariogram(ps[1],ps[2])
    g = 1 - covar(h,vg)
    m = 2.198*median(empvg[i])^2
    vv = 2*g - m
    v += log(length(empvg[i]))*vv*vv
  end
  v
end


#g!(f) = (dp,p) -> dp[:] = ForwardDiff.gradient(f,p)[:]
#h!(f) = (dp,p) -> dp[:,:] = ForwardDiff.hessian(f,p)[:,:]


function fitVario(dat::Data,root::Union{Split,Leaf})
  L = dat.L
  fn = dat.freqNr
  K = div(L,fn)
  empvg = [ Float64[] for i in 1:fn ]
  rsR = zeros(L)
  rsI = zeros(L)
  for i in 1:L
    lf = leaffor(root,dat,i)
    rsR[i] = dat.ResR[i]/lf.p
    rsI[i] = dat.ResI[i]/lf.p
  end
  for k in 1:(K-1)
    for u1 in (k*fn+1):(fn+k*fn)
      for u2 in (u1+1):(fn+k*fn)
        lag = u2-u1+1
        vr = abs(rsR[u1]-rsR[u2])
        vi = abs(rsI[u1]-rsI[u2])
        push!(empvg[lag],vr)
        push!(empvg[lag],vi)
      end
    end
  end
  f(ps) = varioLoss(empvg,dat.dLogFreq,ps)
  op = optimize(f,[0.1,1.1],[100.0,1.9],[1.0,1.5])
  ExpVariogram(op.minimizer...)
end


function bestSplit(dat::Data,us::Vector{Bool},ats::Vector{Int})
  lf = bestLeaf(dat,us)
  ns = Union{Split,Leaf}[lf]
  vs = Float64[lf.v]
  co = log(length(ats))
  for a in ats
    us1,us2 = splitUs(dat,us,a)
    lf1 = bestLeaf(dat,us1)
    lf2 = bestLeaf(dat,us2)
    sp = Split(a,lf1,lf2)
    v = lf1.v + lf2.v
    v += -log(spr()) + co
    sp.v = v
    sp.N = sum(us1) + sum(us2)
    push!(ns,sp)
    push!(vs,v)
  end
  i = argmin(vs)
  ns[i]
end


function bestTree(dat::Data,us::Vector{Bool},ats::Vector{Int})
  n = bestSplit(dat,us,ats)
  if isa(n,Leaf)
    return n
  else
    us1,us2 = splitUs(dat,us,n.con)
    atss = filter(x->x!=n.con,ats)
    n1 = bestTree(dat,us1,atss)
    n2 = bestTree(dat,us2,atss)
    v = n1.v+n2.v-log(spr())+log(length(ats))
    sp = Split(n.con,n1,n2,n1.N+n2.N,v)
    return sp
  end
end


function bestTree(dat::Data)
  ats = [1:dat.D...]
  us = ones(Bool,length(dat))
  n = bestTree(dat,us,ats)
  Tree(n,fitVario(dat,n))
end


function expandTree(dat::Data,tr::Leaf,us::Vector{Bool},ats::Vector{Int},fixed::Bool)
  if fixed
    lf = deepcopy(tr)
    trainLeaf!(dat,us,lf)
    return lf
  else
    return bestTree(dat,us,ats)
  end
end


function expandTree(dat::Data,tr::Split,us::Vector{Bool},ats::Vector{Int},fixed::Bool)
  us1,us2 = splitUs(dat,us,tr.con)
  atss = filter(x->x!=tr.con,ats)
  n1 = expandTree(dat,tr.t1,us1,atss,fixed)
  n2 = expandTree(dat,tr.t2,us2,atss,fixed)
  v = n1.v+n2.v-log(spr())+log(length(ats))
  Split(tr.con,n1,n2,n1.N+n2.N,v)
end


function expandTree(dat::Data,tr::Tree)
  ats = [1:dat.D...]
  us = ones(Bool,length(dat))
  n = expandTree(dat,tr.root,us,ats,false)
  Tree(n,fitVario(dat,n))
end


function expandTree(dat::Data,tr::Tree,ats::Vector{Symbol})
  ats2 = Vector{Int}()
  for s in ats
    i = findfirst(x->x==s,dic.symb)
    if isnothing(i) ; error("attribute $a not recognised!") ; end
    push!(ats2,i)
  end
  us = ones(Bool,length(dat))
  n = expandTree(dat,tr.root,us,ats2,false)
  Tree(n)
end


function fitTree(dat::Data,tr::Tree)
  ats = [1:dat.D...]
  us = ones(Bool,length(dat))
  n = expandTree(dat,tr.root,us,ats,true)
  Tree(n,fitVario(dat,n))
end


objective(tr::Tree,dat::Data) = tr.root.v / length(dat)






