import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm 
import random

from Encoder import Encoder
from ExpertBackbone import FusionExpert
from Decoder import Decoder
from RegionMoFE import RegionMoFE
from parse_args import args
import utils
import tasks_NY.tasks_crime, tasks_NY.tasks_chk, tasks_NY.tasks_serviceCall, tasks_NY.tasks_pop
import tasks_Chi.tasks_crime, tasks_Chi.tasks_chk, tasks_Chi.tasks_serviceCall, tasks_Chi.tasks_pop

seed = args.seed

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

features, mob_adj, poi_sim, land_sim, poi_counts = utils.load_data()

city = args.city
embedding_size = args.embedding_size
d_prime = args.d_prime
c = args.c 
POI_dim = args.POI_dim
landUse_dim = args.landUse_dim
region_num = args.region_num

def _mob_loss(s_embeddings, t_embeddings, mob):
    inner_prod = torch.mm(s_embeddings, t_embeddings.T)
    softmax1 = nn.Softmax(dim=-1)
    phat = softmax1(inner_prod)
    loss = torch.sum(-torch.mul(mob, torch.log(phat + 0.0001)))
    inner_prod = torch.mm(t_embeddings, s_embeddings.T)
    softmax2 = nn.Softmax(dim=-1)
    phat = softmax2(inner_prod)
    loss += torch.sum(-torch.mul(torch.transpose(mob, 0, 1), torch.log(phat + 0.0001)))
    return loss

def _general_loss(embeddings, adj):
    inner_prod = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)
    loss = F.mse_loss(inner_prod, adj)
    return loss

class MultiTaskLoss(nn.Module):
    def __init__(self):
        super(MultiTaskLoss, self).__init__()

    def forward(self, out_s, out_t, mob_adj, out_p, poi_sim, out_l, land_sim):
        mob_loss = _mob_loss(out_s, out_t, mob_adj)
        poi_loss = _general_loss(out_p, poi_sim)
        land_loss = _general_loss(out_l, land_sim)
        loss = poi_loss + land_loss + mob_loss
        return loss
    
class AdaptiveLoss(nn.Module):
    def __init__(self, num_losses):
        super().__init__()
        # learnable weights
        self.weights = nn.Parameter(torch.ones(num_losses))  
        
    def forward(self, losses):
        """
        losses: list of loss tensors [loss1, loss2, ...]
        return: weighted loss + learnable weights
        """
        # calculate current loss ratios (prevent division by zero)
        loss_ratios = [loss.detach() for loss in losses]
        base = sum(loss_ratios) / len(losses) + 1e-8
        ratios = [r / base for r in loss_ratios]
        
        normalized_weights = torch.softmax(self.weights, dim=0)
        
        total_loss = sum(w * loss for w, loss in zip(normalized_weights, losses))
        
        self._current_ratios = ratios  
        return total_loss

def train_model(args, model_kwargs, fusion_backbone):
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    multi_task_loss = MultiTaskLoss()
    adapt_loss = AdaptiveLoss(num_losses=2)
    encoder = Encoder(POI_dim, landUse_dim, region_num, c).to(device)
    decoder = Decoder(model_kwargs["embedding_size"], region_num).to(device)
    model = RegionMoFE(
        num_modalities=args.num_views,
        fusion_model=fusion_backbone,   # TODO：deepcopy
        embedding_size=model_kwargs["embedding_size"],
        hidden_dim=region_num,
        hidden_dim_rw=model_kwargs["hidden_dim_rw"],
        num_layer_rw=model_kwargs["num_layer_rw"],
        temperature_rw=model_kwargs["temperature_rw"],
        triplet_margin=model_kwargs["triplet_margin"]
    ).to(device)

    print(f"Model configuration: {model_kwargs}")

    params = list(model.parameters()) + list(encoder.parameters()) + list(adapt_loss.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(params, lr=model_kwargs["learning_rate"], weight_decay=model_kwargs["weight_decay"])
    best_chk_emb = np.zeros((region_num, embedding_size))
    best_crime_emb = np.zeros((region_num, embedding_size))
    best_serviceCall_emb = np.zeros((region_num, embedding_size))
    best_pop_emb = np.zeros((region_num, embedding_size))
    best_chk_r2 = 0
    best_crime_r2 = 0
    best_serviceCall_r2 = 0
    best_pop_r2 = 0
    chk_interaction_weights = torch.zeros((region_num, 5))
    crime_interaction_weights = torch.zeros((region_num, 5))
    serviceCall_interaction_weights = torch.zeros((region_num, 5))
    pop_interaction_weights = torch.zeros((region_num, 5))

    interaction_losses = {}
    for i in range(args.num_views):
        interaction_losses[f"view_{i+1}"] = []
    interaction_losses[f"syn"] = []
    interaction_losses[f"red"] = []

    # early stop
    epochs = model_kwargs["epochs"]
    isQuit = False  
    patience = 450  
    epochs_without_improvement = 0
      

    for epoch in tqdm(range(epochs), desc="Training"):
        encoder.train()
        model.train()
        isImprove = False

        input_features = encoder(features)
        _, interaction_weights, outputs, interaction_losses = model(input_features)
        out_s, out_t, out_p, out_l= decoder(outputs)

        ### calculate loss
        task_loss = multi_task_loss(out_s, out_t, mob_adj, out_p, poi_sim, out_l, land_sim)
        interaction_loss = sum(interaction_losses) / (args.num_views + 2)

        loss = adapt_loss([task_loss, interaction_loss])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 30 == 0:
            print("Epoch {}, Loss {}, Task loss {}, Interaction loss {}".format(epoch, loss.item(), task_loss.item(), interaction_loss.item()))
            display = True

            embs = outputs
            embs = embs.detach().cpu().numpy()

            if city == "NY":
                _, _, crime_r2 = tasks_NY.tasks_crime.do_tasks(embs, display=display)
                _, _, chk_r2 = tasks_NY.tasks_chk.do_tasks(embs, display=display)
                _, _, serviceCall_r2 = tasks_NY.tasks_serviceCall.do_tasks(embs, display=display)
                _, _, pop_r2 = tasks_NY.tasks_pop.do_tasks(embs, display=display)
            elif city == "Chi":
                _, _, crime_r2 = tasks_Chi.tasks_crime.do_tasks(embs, display=display)
                _, _, chk_r2 = tasks_Chi.tasks_chk.do_tasks(embs, display=display)
                _, _, serviceCall_r2 = tasks_Chi.tasks_serviceCall.do_tasks(embs, display=display)
                _, _, pop_r2 = tasks_Chi.tasks_pop.do_tasks(embs, display=display)

            if best_chk_r2 < chk_r2:
                best_chk_r2 = chk_r2
                best_chk_emb = embs
                chk_interaction_weights = interaction_weights
                isImprove = True
            if best_crime_r2 < crime_r2:
                best_crime_r2 = crime_r2
                best_crime_emb = embs
                crime_interaction_weights = interaction_weights
                isImprove = True
            if best_serviceCall_r2 < serviceCall_r2:
                best_serviceCall_r2 = serviceCall_r2
                best_serviceCall_emb = embs
                serviceCall_interaction_weights = interaction_weights
                isImprove = True
            if best_pop_r2 < pop_r2:
                best_pop_r2 = pop_r2
                best_pop_emb = embs
                pop_interaction_weights = interaction_weights
                isImprove = True
            print(f"Intermediate best: {best_crime_r2}, {best_chk_r2}, {best_serviceCall_r2}, {best_pop_r2}")
            
            if isImprove:
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 30

            if epoch > 300:
                if crime_r2 < 0 or chk_r2 < 0 or serviceCall_r2 < 0 or pop_r2 < 0:
                    isQuit = True
                    break

            np.save(f"./best_emb_checkIn.npy", best_chk_emb)
            np.save(f"./best_emb_crime.npy", best_crime_emb)
            np.save(f"./best_emb_serviceCall.npy", best_serviceCall_emb)
            np.save(f"./best_emb_pop.npy", best_pop_emb)

        if epochs_without_improvement >= patience:
            break

        if isQuit:
            break

    return (crime_interaction_weights, chk_interaction_weights, serviceCall_interaction_weights, pop_interaction_weights)


def test_all_tasks(city):
    best_chk_emb = np.load(f"./best_emb_checkIn.npy")
    best_crime_emb = np.load(f"./best_emb_crime.npy")
    best_serviceCall_emb = np.load(f"./best_emb_serviceCall.npy")
    best_pop_emb = np.load(f"./best_emb_pop.npy")
    
    print("Best region embeddings")
    if city == "NY":
        print('>>>>>>>>>>>>>>>>>   Crime in New York City')
        mae, rmse, crime_r2 = tasks_NY.tasks_crime.do_tasks(best_crime_emb)
        print('>>>>>>>>>>>>>>>>>   Check-In in New York City')
        mae, rmse, chk_r2 = tasks_NY.tasks_chk.do_tasks(best_chk_emb)
        print('>>>>>>>>>>>>>>>>>   Service Calls in New York City')
        mae, rmse, serviceCall_r2 = tasks_NY.tasks_serviceCall.do_tasks(best_serviceCall_emb)
        print('>>>>>>>>>>>>>>>>>   Population in New York City')
        mae, rmse, pop_r2 = tasks_NY.tasks_pop.do_tasks(best_pop_emb)
    elif city == "Chi":
        print('>>>>>>>>>>>>>>>>>   Crime in Chicago')
        mae, rmse, crime_r2 = tasks_Chi.tasks_crime.do_tasks(best_crime_emb)
        print('>>>>>>>>>>>>>>>>>   Check-In in Chicago')
        mae, rmse, chk_r2 = tasks_Chi.tasks_chk.do_tasks(best_chk_emb)
        print('>>>>>>>>>>>>>>>>>   Service Calls in Chicago')
        mae, rmse, serviceCall_r2 = tasks_Chi.tasks_serviceCall.do_tasks(best_serviceCall_emb)
        print('>>>>>>>>>>>>>>>>>   Population in Chicago')
        mae, rmse, pop_r2 = tasks_Chi.tasks_pop.do_tasks(best_pop_emb)

if __name__ == '__main__':
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    if args.city == "NY":
        model_kwargs = {
            ### params for All
            "seed": args.seed,
            "epochs": args.epochs,
            "num_modalities": args.num_views,
            "embedding_size": embedding_size,
            "dropout": args.dropout,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,

            "learning_rate": 0.0005062945957077429,
            "hidden_dim_rw": 256,
            "num_layer_rw": 3,
            "temperature_rw": 0.9,
            "triplet_margin": 0.9,  
            "num_graphormer": 3,
            "no_head": 4,
            "spatial_alpha": 0.5
        }
    if args.city == "Chi":
        model_kwargs = {
            ### params for All
            "seed": args.seed,
            "epochs": args.epochs,
            "num_modalities": args.num_views,
            "embedding_size": embedding_size,
            "dropout": args.dropout,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,

            "learning_rate": 0.0002162169502909867,
            "hidden_dim_rw": 256,
            "num_layer_rw": 3,
            "temperature_rw": 0.4,
            "triplet_margin": 0.1,
            "num_graphormer": 4,
            "no_head": 1,
            "spatial_alpha": 0.4
        }

    print('Model Training-----------------')

    fusion_backbone = FusionExpert(POI_dim, landUse_dim, c,
                                   region_num, 
                                   embedding_size, 
                                   d_prime,
                                   model_kwargs["num_graphormer"], 
                                   model_kwargs["no_head"], 
                                   model_kwargs["dropout"], 
                                   model_kwargs["spatial_alpha"]).to(device)

    crime_weights, chk_weights, serviceCall_weights, pop_weights = train_model(args, model_kwargs, fusion_backbone)


    print("Downstream task test-----------")
    test_all_tasks(city)


        
