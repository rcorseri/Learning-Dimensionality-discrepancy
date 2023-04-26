Learn and save a tree
---------------------


Activate and load the project
'''
]activate .
using MTUL
'''

Create the attribute dictionary: on which elements the tree will
split: look at the function for details
'''
dic = attributeDefs1() ;
'''

Read data for training
specify data file
specify the number of frequencies per site
will use determinants as predictors if last parameter is true 
'''
tData = readTrainingData("data/G_trn10.csv",dic,41,true) ;
'''


Train a tree using 'tree()' as starting tree
Ignore warnings on screen, if any
'''
tr1 = expandTree(tData,tree1(dic)) ;
'''

Optional, create tree pdf
must have 'dot' installed
'''
treePdf(tr1,dic,"tree")
'''

Save tree to file to file, the name must end in '.jld'
'''
saveTree("tree.jld",tr1)
'''


Save tree to file to .txt file
'''
saveTreeTxt(tr1,dic,"tr1.txt")
'''



Load a saved tree
-----------------

Activate and load the project
'''
]activate .
using MTUL
'''

Create the attribute dictionary
must be same as when training
'''
dic = attributeDefs1() ;
'''

Load tree from file, the name must end in '.jld'
'''
tr1 = loadTree("tree.jld")
'''
