"""
	Understanding contents of training data file.
	Note: This is not cleaned up!!!

	# datapoints = 1229838

	# 5, T = 1122012
	human_genes = {'HLA-G', 'HLA-A', 'HLA-E', 'HLA-C', 'HLA-B'}
	num_HLA_G = 2347, num_HLA_A = 347054, num_HLA_E = 38, num_HLA_C = 105559, num_HLA_B = 667014
	# Note: non-classical (E, G) = 2385; classical (A, B, C) = 1119627

	# 11, T = 107826
	non_human_genes = {'SLA-1', 'BoLA-', 'H-2-D', 'Eqca-', 'Mamu-', 'H-2-K', 'Patr-', 'SLA-2', 'H-2-L', 'DLA-8', 'Gogo-'}
	num_SLA_1 = 227, num_BoLA = 157, num_H_2_D = 46727, num_Eqca = 589, 
	num_Mamu = 11839, num_H_2_K = 39517, num_Patr = 3416, num_SLA_2 = 100, 
	num_H_2_L = 752, num_DLA_8 = 4487, num_Gogo = 15

	# 3
	datasets = {'IEDB', 'MHCFlury2_S1', 'S3_Only'}
	
	# 3
	is_ABC_set = {'HLA-B', 'HLA-C', 'HLA-A'}
	# 13 
	not_ABC_set = {'SLA-1', 'BoLA-', 'H-2-D', 'Eqca-', 'Mamu-', 'HLA-G', 'H-2-K', 'Patr-', 'SLA-2', 'HLA-E', 'H-2-L', 'DLA-8', 'Gogo-'}
	# Note: all non_human_genes also have is_ABC == "False"

"""

import csv
from pprint import pprint

def munge_full_train_data():
	i = 0
	human_genes, non_human_genes = set(), set()
	is_human_count, not_human_count = 0, 0
	num_HLA_G, num_HLA_A, num_HLA_E, num_HLA_C, num_HLA_B = 0, 0, 0, 0, 0
	num_SLA_1, num_BoLA, num_H_2_D, num_Eqca, num_Mamu, num_H_2_K, num_Patr, num_SLA_2, num_H_2_L, num_DLA_8, num_Gogo = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

	datasets = set()
	is_ABC_set, not_ABC_set = set(), set()

	non_human_and_not_ABC_count = 0

	with open('mhcglobe_full_train_data.csv', newline='') as csvfile:
	    reader = csv.DictReader(csvfile)
	    for row in reader:
	    	i += 1
	    	# print(row, "\n")
	    	gene, dataset, is_ABC, is_human = row["Gene"], row["dataset"], row["is_ABC"], row["is_human"]
	    	# print(gene, dataset, is_human, is_ABC)

	    	# datasets
	    	datasets.add(dataset)

	    	if is_human == "True":
	    		is_human_count += 1
	    		human_genes.add(gene)
	    		if gene == "HLA-G":
	    			num_HLA_G += 1
	    		elif gene == "HLA-A":
	    			num_HLA_A += 1
	    		elif gene == "HLA-E":
	    			num_HLA_E += 1
	    		elif gene == "HLA-C":
	    			num_HLA_C += 1
	    		elif gene == "HLA-B":
	    			num_HLA_B += 1

	    	else:
	    		not_human_count += 1
	    		non_human_genes.add(gene)
	    		if gene == "SLA-1":
	    			num_SLA_1 += 1
	    		elif gene == "BoLA-":
	    			num_BoLA += 1
	    		elif gene == "H-2-D":
	    			num_H_2_D += 1
	    		elif gene == "Eqca-":
	    			num_Eqca += 1
	    		elif gene == "Mamu-":
	    			num_Mamu += 1
	    		elif gene == "H-2-K":
	    			num_H_2_K += 1
	    		elif gene == "Patr-":
	    			num_Patr += 1
	    		elif gene == "SLA-2":
	    			num_SLA_2 += 1
	    		elif gene == "H-2-L":
	    			num_H_2_L += 1
	    		elif gene == "DLA-8":
	    			num_DLA_8 += 1
	    		elif gene == "Gogo-":
	    			num_Gogo += 1
	    		# Are all non-humans also not_ABC? - Yes!
	    		if is_ABC == "False":
	    			non_human_and_not_ABC_count += 1

	    	# ABC conditions
	    	if is_ABC == "True":
	    		is_ABC_set.add(gene)
	    	else:
	    		not_ABC_set.add(gene)

	print(f"human_genes = {human_genes}")
	print("\n")
	print(f"non_human_genes = {non_human_genes}")
	print("\n")
	print(f"datasets = {datasets}")
	print("\n")
	print(f"is_ABC_set = {is_ABC_set}")
	print("\n")
	print(f"not_ABC_set = {not_ABC_set}")

	print(f"num_HLA_G = {num_HLA_G}, num_HLA_A = {num_HLA_A}, num_HLA_E = {num_HLA_E}, num_HLA_C = {num_HLA_C}, num_HLA_B = {num_HLA_B}")
	print(f"num_SLA_1 = {num_SLA_1}, num_BoLA = {num_BoLA}, num_H_2_D = {num_H_2_D}, num_Eqca = {num_Eqca}, num_Mamu = {num_Mamu}, num_H_2_K = {num_H_2_K}, num_Patr = {num_Patr}, num_SLA_2 = {num_SLA_2}, num_H_2_L = {num_H_2_L}, num_DLA_8 = {num_DLA_8}, num_Gogo = {num_Gogo}")

	print("\n", non_human_and_not_ABC_count == not_human_count)
    	
if __name__ == "__main__":
	munge_full_train_data()
 
