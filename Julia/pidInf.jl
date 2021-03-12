using Pkg
using NetworkInference
using LightGraphs
using GraphPlot

#datapath = "/nfs/numerik/araharin/Biological_Switches/Julia/Tests/"
#dataset_name = string(datapath,"simulated_datasets/", 50, "_", "ecoli1_large", ".txt")

datapath = "/nfs/numerik/araharin/Biological_Switches/mRNA_Sim/txt/"
#dataset_name = string(datapath,"two_MRNA_Double_Up_data.txt")
#dataset_name = string(datapath,"two_MRNA_No_Up_data.txt")
dataset_name = string(datapath,"two_MRNA_Single_Up_data.txt")


#algorithm = PIDCNetworkInference() # minimum 3 species
#algorithm = PUCNetworkInference() # minimum 3 species
#algorithm = CLRNetworkInference() # minimum ?
algorithm = MINetworkInference() # minimum 2 
# Keep the top x% highest-scoring edges
# 0.0 < threshold < 1.0
#threshold = 0.15 # keep top (1 - threshold)*100 percent highest-scoring edges

@time genes = get_nodes(dataset_name)
@time network = InferredNetwork(algorithm, genes)

#adjacency_matrix, labels_to_ids, ids_to_labels = get_adjacency_matrix(network, threshold)
#graph = LightGraphs.SimpleGraphs.SimpleGraph(adjacency_matrix)

#number_of_nodes = size(adjacency_matrix)[1]
#nodelabels = []
#for i in 1 : number_of_nodes
           #push!(nodelabels, ids_to_labels[i])
       #end

println(dataset_name)
for edge in network.edges
       println(edge.nodes[1].label," ", edge.nodes[2].label," ",  edge.weight)
       end
#println(adjacency_matrix)      
#using Cairo, Compose
#draw(PNG("Double_up.png"),gplot(graph, nodelabel = nodelabels))
#draw(PNG("No_up.png"),gplot(graph, nodelabel = nodelabels))
#draw(PNG("Single_up.png"),gplot(graph, nodelabel = nodelabels))

#draw(PNG("test.png"),gplot(graph, nodelabel = nodelabels))

#draw(PDF("test.pdf"),gplot(graph, nodelabel = nodelabels))
#draw(SVG("test.svg"),gplot(graph, nodelabel = nodelabels))
