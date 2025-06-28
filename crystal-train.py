from loaders.crystal_dataset import CrystalGraphDataset
from torch.utils.data import Dataset, DataLoader, random_split
from loaders.collate import crystal_collate
import torch
from models.model_loader import load_model
import torch.optim as optim
from models.losses import GradientConsistencyLoss
import torch.nn as nn
from torch_geometric.data import Batch
from configs.config import Config
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR

import argparse
from torch.utils.data import Subset

"""
python crystal-train.py -n Relaxation_Large -p 100 -m 1

Next steps: 

python crystal-train.py -n Bulk -p 100 -m 1


1. Change this to Relaxation_Large
2. Change this to Relaxation_fully
3. Add another command that lets me use the complicated one.

python crystal-train.py -n Relaxation_Large
python crystal-train.py -n Relaxation_Fully
python crystal-train.py -n Relaxation_Sepearted -g 2
"""


def train(config, dataloader_train, dataloader_val, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(config, int(args.multiplier), int(args.graphNum)).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print("Total Params: {}".format(total_params))

    
    
    optimizer = optim.Adam(model.parameters(), lr=config.training.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.training.n_epochs)
    criterion_mse = nn.L1Loss()
    lambda_gradient = config.training.lambda_gradient
    min_val = 10000.0

    for epoch in range(config.training.n_epochs):
        total_tr_loss = 0
        model.train()
        for batch_idx, (
            obj_name,
            soft_rest_graphs,
            soft_def_graphs,
            rigid_graphs,
            _,
        ) in enumerate(dataloader_train):
            soft_rest_graphs_batched = soft_rest_graphs
            soft_def_graphs_batched = soft_def_graphs


            soft_rest_graphs_batched, soft_def_graphs_batched = (
                soft_rest_graphs_batched.to(device),
                soft_def_graphs_batched.to(device),
            )


            if(soft_rest_graphs_batched.x.size() == torch.Size([0])):
                continue

            if(int(args.graphNum) == 2):
                rigid_graphs = rigid_graphs.to(device)
            
            if(int(args.graphNum) == 1):

                predictions = model(soft_rest_graphs_batched)
            else:
                predictions = model(soft_rest_graphs_batched, rigid_graphs)

            predictions.pos = predictions.pos - soft_rest_graphs_batched.pos
            soft_def_graphs_batched.pos = (
                soft_def_graphs_batched.pos - soft_rest_graphs_batched.pos
            )



            loss_mae = criterion_mse(predictions.pos, soft_def_graphs_batched.pos)

            loss_pos = criterion_mse(predictions.pos, soft_def_graphs_batched.pos)
            loss_neg = criterion_mse(predictions.pos, soft_rest_graphs_batched.pos)
            loss_triplet = (loss_pos + 0.001) / (loss_neg + 0.001)
            tr_loss = loss_mae
            
            total_tr_loss += tr_loss.item()
            optimizer.zero_grad()
            tr_loss.backward()
            optimizer.step()
        
        scheduler.step()

        # Validation loop
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (
                obj_name,
                soft_rest_graphs,
                soft_def_graphs,
                rigid_graphs,
                _,
                _,
            ) in enumerate(dataloader_val):
                soft_rest_graphs_batched = soft_rest_graphs
                soft_def_graphs_batched = soft_def_graphs
                (
                    soft_rest_graphs_batched,
                    soft_def_graphs_batched,
                ) = (
                    soft_rest_graphs_batched.to(device),
                    soft_def_graphs_batched.to(device),
                )
                if(soft_rest_graphs_batched.x.size() == torch.Size([0])):
                    continue


                if(int(args.graphNum) == 2):
                    rigid_graphs = rigid_graphs.to(device)

                if(int(args.graphNum) == 1):

                    predictions = model(soft_rest_graphs_batched)
                else:
                    predictions = model(soft_rest_graphs_batched, rigid_graphs)


                loss_mae = criterion_mse(predictions.pos, soft_def_graphs_batched.pos)
                # loss_deformable = criterion_def(predictions, soft_def_graphs_batched, meta_data['deform_intensity'].to(device))
                loss_val = loss_mae

                total_val_loss += loss_val.item()

        avg_val_loss = total_val_loss / len(dataloader_val)

        avg_tr_loss = total_tr_loss / len(dataloader_train)
        print(
            f"Epoch {epoch+1}/{config.training.n_epochs} - Training Loss: {avg_tr_loss} - Validation Loss: {avg_val_loss}"
        )

        # Logging validation loss to wandb
        print({"validation_loss": avg_val_loss})
    
    torch.save(model, "model.pt")
    print("!Results - validation_loss: {}, Params: {}, samples: {}".format(avg_val_loss, total_params, args.proportion))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Help msg", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-n", "--name",  help="name",  required=True)
    parser.add_argument("-p", "--proportion", help="how many samples do you want", default=-1)
    parser.add_argument("-m", "--multiplier", help="How much you want to scale up the model", default=1)
    parser.add_argument("-g", "--graphNum", help="How many graphs we want", default=1)

    args = parser.parse_args()


    dataset = CrystalGraphDataset(args.name, graphNum = args.graphNum)

    batch_size = 32
    proportion = int(args.proportion)


    #Okay, so idea number one failed; the tallest element isn't always the adosrbate.

    # I can check to see if every element falls into one of these:

    # There is 1 hydrogen, 1 carbon, 1 oxygen.

    # what irriates me above all else is the idea that there could be one with multiple carbons, multiple oxygens, and 0 Hydrogens.

    # Okay, screw this. I can't handle this dudes. It's way to slow to process everything here

    if(proportion == -1):

        split = random_split(dataset, [0.8, 0.2])
        train_dataset = split[0]
        test_dataset = split[1]
        args.proportion = len(train_dataset)
    else:
        train_dataset = Subset(dataset, range(proportion))
        test_dataset = Subset(dataset, range(proportion))
    # alright, what do I want to do?
    # find the tallest one.
    # Check the identity.
    # If H, leave.
    # If O or C, then find the 2nd highest.
    # What's the easiest way? May be to duplicate. Have some sort of keyword.
    # No, I don't want to deal with that. 
    # Just process the whole dataset right here, before I toss it into the loader. Good.
    # Then toy with the collate function. 

    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=crystal_collate)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, collate_fn=crystal_collate)


    config_path = "configs/everyday.json"
    config = Config(config_path)
    print("Begin Training")
    train(config, train_loader, test_loader, args)
