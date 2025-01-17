# getitem must return: None, input_graph, target_graph, null, null.
import os
from torch.utils.data import Dataset
import torch
from torch_geometric.data import Data



class CrystalGraphDataset(Dataset):
    def __init__(self, datasetName, edge_generation="all"):
        super().__init__()
        self.dataset_path = "dataset"
        self.raw_path = os.path.join(self.dataset_path, "raw")
        self.processed_path = os.path.join(self.dataset_path, "processed")

        self.dataset_dir = os.path.join(self.processed_path, datasetName)


        if(os.path.exists(self.dataset_dir)):
            self.len = len(os.listdir(self.dataset_dir))
            print("Processings skipped because diectory {} exists. If you wish to process raw data, please delete that folder.".format(datasetName))
        else:
            self.process(datasetName, edge_generation)
    
    def __getitem__(self, idx):
        filename = os.path.join(self.dataset_dir, "Element_{}.pt".format(idx))
        item = torch.load(filename)

        # items are both Data objects representing the input and target graphs respectively.
        return None, item[0], item[1], None, None

    def __len__(self):
        return self.len
            
           
    def process(self,datasetName, edge_generation):
        os.mkdir(self.dataset_dir)
        
        
        raw_file = open(os.path.join(self.raw_path, "{}.txt".format(datasetName)), mode='r')

        num_nodes = raw_file.readline()
        idx = 0
        while(not(num_nodes == "")):
            num_nodes = int(num_nodes)
            start_pos_list = []
            end_pos_list = []
            features_list = []
            elements = []
            edges = []
            filename = os.path.join(self.dataset_dir, "Element_{}.pt".format(idx))

            # read data
            for i in range(num_nodes):
                node_line = raw_file.readline()
                parts = node_line.split(":")
                elements.append(parts[0])
                start_pos_list.append([float(feature) for feature in parts[2].split(",")])
                end_pos_list.append([float(feature) for feature in parts[3].split(",")])

                features_list.append([float(p) for p in parts[1].split(",")])
            

            if (edge_generation == "all"):
               for i in range(num_nodes):
                   for j in range(num_nodes): 
                       
                       if(i != j):
                           edges.append([i, j])
            else:
                # otherwise, have an empty edge connection
                pass
            
            edges = torch.tensor(edges)
            edges = torch.transpose(edges, 0, -1)
            start_pos_list = torch.tensor(start_pos_list)
            features_list = torch.tensor(features_list)
            end_pos_list = torch.tensor(end_pos_list)


            input_graph = Data(x=features_list, edge_index=edges, pos=start_pos_list,elements=elements)
            target_graph = Data(x=features_list, edge_index=edges, pos=end_pos_list)

            data = (input_graph, target_graph)

            torch.save(data, filename)
        
            num_nodes = raw_file.readline()
            idx += 1
    
        self.len = idx


if __name__ == "__main__":
    dataset = CrystalGraphDataset("Random")