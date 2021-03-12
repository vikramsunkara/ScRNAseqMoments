using Pkg
using NetworkInference
using LightGraphs
using GraphPlot
using CSV
using DataFrames

datapath = "/nfs/datanumerik/people/araharin/Data_032021/Unpacked_MRNA_data_10000/txt_start30/"

for s in 4:5
    path_kernels = [string("Double_Up_data_", s-1, '_'), string("Double_Up_data_", s-1, "_1chng_"), string("Single_Up_data_", s-1, '_')] # Check order of data in result dataframe
    
    algorithm = MINetworkInference()
    
    collect_MI_kernels = []
    collect_data_num_kernels = []
    collect_nodes_1_kernels = []
    collect_nodes_2_kernels = []
    collect_edges_num = []
    
    for i in 1:size(path_kernels, 1)
    	kernel = path_kernels[i]
    	collect_MI = []
    	collect_data_num = []
    	collect_nodes_1 = []
    	collect_nodes_2 = []
    	collect_num = []
    	for j in 1:400
    		dataset_name = string(datapath, kernel, j-1, ".txt")
    		@time genes = get_nodes(dataset_name)
    		@time network = InferredNetwork(algorithm, genes)
    		numEdges = size(network.edges, 1)
    		println(dataset_name, " ", "num edges ", " ", numEdges)
    		#if numEdges == 1  # checked that there is not more than one inferred edge
    			#println(" ")
    			#for edge in network.edges
    				#println(edge.nodes[1].label, " ", edge.nodes[2].label, " ", edge.weight)
    			#end
    		#else
    			#println("There are more than one Edges")
    		for edge in network.edges 
    			append!(collect_MI, [edge.weight])
    			append!(collect_data_num, [j-1])
    			append!(collect_nodes_1, [edge.nodes[1].label])
    			append!(collect_nodes_2, [edge.nodes[2].label])
    			end
    		append!(collect_num, [numEdges])
    		end
    	push!(collect_MI_kernels, collect_MI)
    	push!(collect_data_num_kernels, collect_data_num)
    	push!(collect_nodes_1_kernels, collect_nodes_1)
    	push!(collect_nodes_2_kernels, collect_nodes_2)
    	push!(collect_edges_num, collect_num)
    	#println(collect_MI)
    	end
    
    #df = DataFrame(Double_Up_num = collect_data_num_kernels[1], Double_Up_node_1 = collect_nodes_1_kernels[1], 
    #		Double_Up_node_2 = collect_nodes_2_kernels[1], Double_Up_MI = collect_MI_kernels[1],
    #
    #		No_Up_num  = collect_data_num_kernels[2], No_Up_node_1 = collect_nodes_1_kernels[2],
    #		No_Up_node_2 = collect_nodes_2_kernels[2], No_Up_MI = collect_MI_kernels[2],
    #
    #		Single_Up_num = collect_data_num_kernels[3], Single_Up_node_1 = collect_nodes_1_kernels[3],
    #		Single_Up_node_2 = collect_nodes_2_kernels[3], Single_Up_MI = collect_MI_kernels[3]
    #		)
    
    df = DataFrame(Double_Up = collect_MI_kernels[1], Double_Up_1chng = collect_MI_kernels[2], Single_Up = collect_MI_kernels[3], 
    		Num_D = collect_edges_num[1], Num_N = collect_edges_num[2], Num_S = collect_edges_num[3])
    
    Result_file = string("MI_10000traj_shift30_", s-1, ".csv")
    CSV.write(Result_file, df)
    end
