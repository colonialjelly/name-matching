import numpy as np
import random
import torch
import torch.nn as nn
from collections import defaultdict
from tqdm import tqdm, trange
from matchers import utils, metrics

def get_near_negatives(input_names, weighted_relevant_names, all_candidates, k=50):
    all_names_unpadded = set([utils.remove_padding(name) for name in (input_names + all_candidates.tolist())])
    near_negatives = defaultdict(list)
    for name, positives in tqdm(zip(input_names, weighted_relevant_names), total=len(input_names)):
        positive_names = set(utils.remove_padding(n) for n, _, _ in positives)
        near_negatives[name] = [utils.add_padding(n) for n in 
                                utils.get_k_near_negatives(utils.remove_padding(name), positive_names, all_names_unpadded, k)]
    return near_negatives


class TripletDataLoader:
    def __init__(self, input_names, weighted_relevant_names, near_negatives, char_to_idx_map, max_name_length, batch_size, shuffle):
        name_pairs = []
        for input_name, positives in zip(input_names, weighted_relevant_names):
            for pos_name, _, _ in positives:
                name_pairs.append([input_name, pos_name])
        self.name_pairs = np.array(name_pairs)
        self.near_negatives = near_negatives
        self.char_to_idx_map = char_to_idx_map
        self.max_name_length = max_name_length
        self.batch_size = batch_size
        self.shuffle = shuffle
        
    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.name_pairs)
        self.ix = 0
        return self

    def __next__(self):
        if self.ix >= self.name_pairs.shape[0]:
            raise StopIteration
        # return a batch of input_names, pos_names, neg_names
        input_names = self.name_pairs[self.ix:self.ix+self.batch_size,0]
        pos_names = self.name_pairs[self.ix:self.ix+self.batch_size,1]
        neg_names = np.apply_along_axis(lambda row: np.array(random.choice(self.near_negatives[row[0]]), object), 1, input_names.reshape(-1,1))
        # not very efficient to re-convert over and over, but it's convenient to do it here
        input_tensor = utils.names_to_one_hot(input_names, self.char_to_idx_map, self.max_name_length)
        pos_tensor = utils.names_to_one_hot(pos_names, self.char_to_idx_map, self.max_name_length)
        neg_tensor = utils.names_to_one_hot(neg_names, self.char_to_idx_map, self.max_name_length)
        self.ix += self.batch_size
        return input_tensor, pos_tensor, neg_tensor
    

def train_triplet_loss(model, input_names, weighted_relevant_names, near_negatives, 
                       input_names_test, weighted_relevant_names_test, candidates_test, candidates_train, all_candidates,
                       char_to_idx_map, max_name_length, num_epochs=100, batch_size=128, margin=0.1, k=100, device="cpu"):
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = torch.nn.TripletMarginLoss(margin=margin, p=2)

    X_train_inputs, _ = utils.convert_names_model_inputs(candidates_train, char_to_idx_map, max_name_length)
    X_test_inputs, _ = utils.convert_names_model_inputs(input_names_test, char_to_idx_map, max_name_length)
    X_test_candidate_inputs, _ = utils.convert_names_model_inputs(candidates_test, char_to_idx_map, max_name_length)

    data_loader = TripletDataLoader(input_names, weighted_relevant_names, near_negatives, char_to_idx_map, max_name_length, batch_size, shuffle=True)

    with trange(num_epochs) as pbar:
        for _ in pbar:
            # train on gpu if there is one
            model.train()
            model.to(device)
            model.device=device
            losses = []
            for i, (anchor_tensor, pos_tensor, neg_tensor) in enumerate(data_loader):
                # Clear out gradient
                model.zero_grad()

                # Compute forward pass
                anchor_emb = model(anchor_tensor, just_encoder=True)
                pos_emb = model(pos_tensor, just_encoder=True)
                neg_emb = model(neg_tensor, just_encoder=True)
                
                loss = loss_fn(anchor_emb, pos_emb, neg_emb)
                losses.append(loss.detach().item())
                loss.backward()
                optimizer.step()

            # eval on cpu because test dataset doesn't fit on gpu
            model.eval()
            model.to("cpu")
            model.device="cpu"
            with torch.no_grad():
                X_train_candidates_encoded = model(X_train_inputs, just_encoder=True).detach().numpy()                
                X_input_names_encoded = model(X_test_inputs, just_encoder=True).detach().numpy()
                X_test_candidates_encoded = model(X_test_candidate_inputs, just_encoder=True).detach().numpy()
                X_candidates_encoded = np.vstack((X_train_candidates_encoded, X_test_candidates_encoded))
                candidates = utils.get_candidates_batch(X_input_names_encoded, 
                                                        X_candidates_encoded, 
                                                        all_candidates,
                                                        num_candidates=k)
            # calc test AUC
            auc = metrics.get_auc(weighted_relevant_names_test, candidates, step=.001)
            print("test AUC", auc)

            # Update loss value on progress bar
            pbar.set_postfix({'loss': sum(losses)/len(losses), 'auc': auc})
                  