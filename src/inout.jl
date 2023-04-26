

## commented out so we don't need to use the plotting library
## ----------------------------------------------------------

# using Plots

#function plotRhoAppSTD(dat::Data,tr::Tree,outfile::String,site::String,diagonalOnly::Bool=false)
#  r = filter(x->dat.site[x]==site,[1:dat.L...])
#  fn = length(r)
#  C,ss = getC(dat,tr,site,diagonalOnly)
#  df = DataFrame([dat.freq[r],dat.PrR[r],dat.PrI[r],ss],[:freq,:Zr,:Zi,:std])#
#
#  plot(1 ./df.freq ,df.Zr,grid=true,yerror=std,xaxis=:log, yaxis=:log)
#end


#function plotSamplesRe(dat::Data,tr::Tree,i::Int)
#  z = R(i)
#  es = getAtt(dat,:elip1,z)
#  j1 = findfirst(x->x,es)
#  if j1 == nothing ; j1 = F()+1 ; end
#  es = getAtt(dat,:beta3,z)
#  j2 = findfirst(x->x,es)
#  if j2 == nothing ; j2 = F()+1 ; end
#  p1 = plot(dat.PrR[z],color=:red,legend=:none)
#  plot!([j1],linetype=:vline,label="elip>0.1",color=:blue)
#  plot!([j2],linetype=:vline,label="beta>3",color=:blue)
#  for j in 1:1000
#    zs1,zs2,ss = sampleLine(i,dat,tr) ;
#    plot!(zs1,legend=:none,color=:grey,alpha=0.1)
#  end
#  plot!(dat.Zxyr[z],color=:yellow,legend=:none)
#  plot!(dat.Zyxr[z],color=:yellow,legend=:none)
#  plot!(dat.PrR[z],color=:red,legend=:none)
#  plot!(dat.Zr[z],color=:green,legend=:none)
#  p1
#end

## ----------------------------------------------------------


function exportResidualData(dat::Data,tr::Tree,outfile::String;addNoise::Bool=false)
  df = DataFrame()
  df[!,:site] = dat.site
  df[!,:freq] = dat.freq
  df[!,:PrR] = dat.PrR
  df[!,:PrI] = dat.PrI
  df[!,:ResR] = dat.ResR
  df[!,:ResI] = dat.ResI
  f(i) = addNoise ? sqrt(leaffor(tr,dat,i).p^2 + dat.noise[i]^2) : leaffor(tr,dat,i).p
  stds = [ f(i) for i in 1:length(dat) ]
  df[!,:std] = stds
  CSV.write(outfile,df,delim=',')
  ;
end


function randLines(dat::Data,tr::Tree,outpath::String,site::String,lines::Int)
  r = filter(x->dat.site[x]==site,[1:dat.L...])
  fn = length(r)
  C,ss = getC(dat,tr,site)
  df = DataFrame(Float64,lines*2,fn)
  dis = MultivariateNormal(zeros(fn),C)
  r = filter(x->dat.site[x]==site,[1:dat.L...])
  for i in 1:lines
    zs1 = rand(dis)
    zs2 = rand(dis)
    for j in 1:fn
      df[(i-1)*2+1,j] = zs1[j] + dat.PrR[r[j]]
      df[(i-1)*2+2,j] = zs2[j] + dat.PrI[r[j]]
    end
  end
  CSV.write(outpath,df)
  ;
end


function nodelist!(dc::Dict{Union{Split,Leaf},String},n::Leaf)
  dc[n] = "A$(count(x->isa(x,Leaf),keys(dc)))"
  ;
end

function nodelist!(dc::Dict{Union{Split,Leaf},String},n::Split)
  dc[n] = "a$(count(x->isa(x,Split),keys(dc)))"
  nodelist!(dc,n.t1)
  nodelist!(dc,n.t2)
  ;
end


function treeFile(t::Tree,dic::AttributeDefs,path::String)
  f = open(path,"w")
  println(f,"digraph MODTREE {")
  lls = Dict{Union{Split,Leaf},String}()
  nodelist!(lls,t.root)
  for k in keys(lls)
    if isa(k,Leaf)
      nn = "$(round(k.p,digits=6))"
      println(f,"  $(lls[k]) [shape=box,label=\"$(nn)\",fontsize=10];")
    else
      println(f,"  $(lls[k]) [label=\"$(dic.name[k.con])\"];")
    end
  end
  c1,c2 = "green","red"
  for k in keys(lls)
    if isa(k,Leaf) ; continue ; end
    println(f,"  $(lls[k]) -> $(lls[k.t1]) [color=$(c1),fontsize=14];")
    println(f,"  $(lls[k]) -> $(lls[k.t2]) [color=$(c2),fontsize=14];")
  end
  println(f,"}")
  close(f)
end


function treePdf(t::Tree,dic::AttributeDefs,path::String)
  pth = deepcopy(path)
  if endswith(pth,".pdf") ; pth = pth[1:(length(pth)-4)] ; end
  if endswith(pth,".gv") ; pth = pth[1:(length(pth)-3)] ; end
  fn1 = pth*".gv"
  fn2 = pth*".pdf"
  treeFile(t,dic,pth*".gv")
  com = `dot -Tps $fn1 -o $fn2`
  run(com)
  sleep(2)
  #com = `rm -f $fn1`
  run(com)
  ;
end



function showT(t::Leaf,off::Int,dic::AttributeDefs)
  p = round(t.p,digits=6)
  v = round(t.v,digits=2)
  #println(" "^off*"Leaf($p,$(t.N),$v)")
  #println(" "^off*"Leaf($p)")
  println("$off Leaf $p ")
end

function showT(t::Split,off::Int,dic::AttributeDefs)
  v = round(t.v,digits=6)
  #println(" "^off*"Split: $(t.con)") 
  #println("  "^off*"Split: $(dic.name[t.con]), $(t.N), $v")
  println("$off Split $(t.con)") 
  showT(t.t1,off+1,dic)
  showT(t.t2,off+1,dic)
end

function showT(t::Tree,dic::AttributeDefs)
  #code to print variogram t.vg
  println("$(t.vg.ex)")
  println("$(t.vg.rho)")
  showT(t.root,0,dic)
end


function saveTreeTxt(t::Leaf,off::Int,dic::AttributeDefs,f)
  p = round(t.p,digits=6)
  v = round(t.v,digits=2)
  #println(" "^off*"Leaf($p,$(t.N),$v)")
  #println(" "^off*"Leaf($p)")
  println(f,"$off Leaf $p ")
end


function saveTreeTxt(t::Split,off::Int,dic::AttributeDefs,f)
  v = round(t.v,digits=6)
  #println(" "^off*"Split: $(t.con)") 
  #println("  "^off*"Split: $(dic.name[t.con]), $(t.N), $v")
  println(f,"$off Split $(t.con)") 
  saveTreeTxt(t.t1,off+1,dic,f)
  saveTreeTxt(t.t2,off+1,dic,f)
end


function saveTreeTxt(t::Tree,dic::AttributeDefs,path::String)
  f = open(path,"w")
  #code to print variogram t.vg
  println(f,"$(t.vg.ex)")
  println(f,"$(t.vg.rho)")
  saveTreeTxt(t.root,0,dic,f)
  close(f)
end



