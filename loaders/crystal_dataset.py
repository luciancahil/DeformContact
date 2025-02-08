# getitem must return: None, input_graph, target_graph, null, null.
import os
from torch.utils.data import Dataset
import torch
from torch_geometric.data import Data



class CrystalGraphDataset(Dataset):
    def __init__(self, datasetName):
        super().__init__()
        self.dataset_path = "dataset"
        self.raw_path = os.path.join(self.dataset_path, "raw")
        self.processed_path = os.path.join(self.dataset_path, "processed")

        self.dataset_dir = os.path.join(self.processed_path, datasetName)


        if(os.path.exists(self.dataset_dir)):
            self.len = len(os.listdir(self.dataset_dir))
            self.elm_to_num = torch.load(os.path.join(self.dataset_dir, "elm_to_num.pt"))
            print("Processings skipped because diectory dataset/processed/{} exists. If you wish to process raw data, please delete that folder.".format(datasetName))
        else:
            self.process(datasetName)
    
    def __getitem__(self, idx):
        filename = os.path.join(self.dataset_dir, "Element_{}.pt".format(idx))
        item = torch.load(filename)

        # items are both Data objects representing the input and target graphs respectively.
        return None, item[0], item[1], None, None

    def __len__(self):
        return self.len
            
           
    def process(self,datasetName):
        os.mkdir(self.dataset_dir)
        
        
        raw_file = open(os.path.join(self.raw_path, "{}.txt".format(datasetName)), mode='r')

        num_nodes = raw_file.readline()
        idx = 0

        all_starting = torch.empty(0, 3)

        all_ending = torch.empty(0, 3)


        element_set = set()


        while(not(num_nodes == "" or num_nodes == "\n")):
            num_nodes = int(num_nodes)
            start_pos_list = []
            end_pos_list = []
            features_list = []
            elements = []
            edges = []
            edge_attr = torch.empty(0, 3)
            filename = os.path.join(self.dataset_dir, "Element_{}.pt".format(idx))

            # read data
            for i in range(num_nodes):
                node_line = raw_file.readline()
                parts = node_line.split(":")
                elements.append(parts[0])
                element_set.add(parts[0])
                start_pos_list.append([float(feature) for feature in parts[2].split(",")])
                end_pos_list.append([float(feature) for feature in parts[3].split(",")])

                features_list.append([float(p) for p in parts[1].split(",")])

                cur_edges = parts[4].strip().split(";")

                for edge in cur_edges:

                    edge_parts = edge.split(",")
                    if(edge_parts[0] == ""):
                        continue
                    edge_target = int(edge_parts[0])
                    edge_data = torch.tensor([float(num) for num in edge_parts[1:]][4:])
                    edge_data = edge_data / torch.norm(edge_data)

                    edges.append([i, edge_target])
                    edge_attr = torch.concat((edge_attr, edge_data.unsqueeze(0)), dim = 0)
            
            edges = torch.tensor(edges)
            edges = torch.transpose(edges, 0, -1)
            edges = edges.long()
            start_pos_list = torch.tensor(start_pos_list)
            features_list = torch.tensor(features_list)
            end_pos_list = torch.tensor(end_pos_list)
            
            all_starting = torch.concat((all_starting, start_pos_list), dim=0)
            all_ending = torch.concat((all_ending, end_pos_list), dim=0)


            input_graph = Data(x=features_list, edge_index=edges, edge_attr=edge_attr, pos=start_pos_list, elements=elements)
            target_graph = Data(x=features_list, edge_index=edges, pos=end_pos_list)

            data = (input_graph, target_graph)

            torch.save(data, filename)
        
            num_nodes = raw_file.readline()
            idx += 1


        self.len = idx
        self.xs_mean = all_starting[:,1].mean()
        self.xs_std = all_starting[:,1].std()
        self.ys_mean = all_starting[:,1].mean()
        self.ys_std = all_starting[:,1].std()
        self.zs_mean = all_starting[:,1].mean()
        self.zs_std = all_starting[:,1].std()

        self.xe_mean = all_ending[:,1].mean()
        self.xe_std = all_ending[:,1].std()
        self.ye_mean = all_ending[:,1].mean()
        self.ye_std = all_ending[:,1].std()
        self.ze_mean = all_ending[:,1].mean()
        self.ze_std = all_ending[:,1].std()

        self.elm_to_num = dict()
        self.num_to_elm = dict()

        for i, element in enumerate(element_set):
            self.elm_to_num[element] = i
            self.num_to_elm[i] = element
        
        # replace the x with the element, and normalize

        for i in range(self.len):
            filename = os.path.join(self.dataset_dir, "Element_{}.pt".format(i))

            item = torch.load(filename)

            numbers = torch.tensor([self.elm_to_num[elm] for elm in item[0].elements]).unsqueeze(1)
            item[0].x = numbers

            item[0].pos[:,0] = (item[0].pos[:,0] - self.xs_mean)/self.xs_std
            item[0].pos[:,1] = (item[0].pos[:,1] - self.ys_mean)/self.ys_std
            item[0].pos[:,2] = (item[0].pos[:,2] - self.zs_mean)/self.zs_std

            item[1].pos[:,0] = (item[0].pos[:,0] - self.xe_mean)/self.xe_std
            item[1].pos[:,1] = (item[0].pos[:,1] - self.ye_mean)/self.ye_std
            item[1].pos[:,2] = (item[0].pos[:,2] - self.ze_mean)/self.ze_std


            torch.save(item, filename)

        torch.save(self.elm_to_num, os.path.join(self.dataset_dir, "elm_to_num.pt"))

if __name__ == "__main__":
    dataset = CrystalGraphDataset("Random")