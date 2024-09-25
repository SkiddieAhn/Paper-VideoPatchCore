import torch
from torch import tensor
from sklearn import random_projection


def get_coreset(
        memory_bank: tensor,
        l: int = 1000,  # Coreset target
        eps: float = 0.09,
        r_proj: bool = False,
        log: bool = False,
        device: str = 'cuda'
    ) -> tensor:
    """
        Returns l coreset indexes for given memory_bank.

        Args:
        - memory_bank:     Patchcore memory bank tensor
        - l:               Number of patches to select
        - eps:             Sparse Random Projector parameter

        Returns:
        - coreset indexes
    """

    coreset_idx = []  # Returned coreset indexes
    idx = 0

    # Fitting random projections
    if r_proj:
        try:
            transformer = random_projection.SparseRandomProjection(eps=eps)
            memory_bank = torch.tensor(transformer.fit_transform(memory_bank))
        except ValueError:
            print("Error: could not project vectors. Please increase `eps`.")

    # Coreset subsampling
    last_item = memory_bank[idx: idx + 1]   # First patch selected = patch on top of memory bank
    coreset_idx.append(torch.tensor(idx))
    min_distances = torch.linalg.norm(memory_bank - last_item, dim=1, keepdims=True)    # Norm l2 of distances (tensor)

    # Use GPU if possible
    if torch.cuda.is_available():
        last_item = last_item.to(device)
        memory_bank = memory_bank.to(device)
        min_distances = min_distances.to(device)

    for l_idx in range(l - 1):
        distances = torch.linalg.norm(memory_bank - last_item, dim=1, keepdims=True)    # L2 norm of distances (tensor)
        min_distances = torch.minimum(distances, min_distances)                         # Verical tensor of minimum norms
        # min_distances = np.minimum(distances.cpu().detach().numpy(), min_distances.cpu().detach().numpy())
        # min_distances = torch.from_numpy(min_distances)
        idx = torch.argmax(min_distances)                                               # Index of maximum related to the minimum of norms

        last_item = memory_bank[idx: idx + 1]   # last_item = maximum patch just found
        min_distances[idx] = 0                  # Zeroing last_item distances
        coreset_idx.append(idx.to("cpu"))       # Save idx inside the coreset
        if log:
            if (l_idx+1) % 1000 == 0:
                print(f'[{l_idx+1}/{l-1}] coreset record ok!')

    return torch.stack(coreset_idx)


def predict(features, memory_bank, power_n=2):
    distances = torch.cdist(features, memory_bank, p=2.0)         # L2 norm dist btw test patch with each patch of memory bank
    dist_score, _ = torch.min(distances, dim=1)       # Val and index of the distance scores (minimum values of each row in distances)
    s_star = torch.max(dist_score)  
    output = s_star ** power_n
    return output


def find_neareset(features, memory_bank):
    distances = torch.cdist(features, memory_bank, p=2.0)        
    dist_score, dist_score_idxs = torch.min(distances, dim=1)  
    m_idx = torch.argmax(dist_score) 
    max_dist = torch.max(dist_score)  

    max_patch = features[m_idx]      
    nearest = memory_bank[dist_score_idxs[m_idx]]
    return max_dist, max_patch, nearest


def predict_all_patches(features, memory_bank):
    distances = torch.cdist(features, memory_bank, p=2.0)         # L2 norm dist btw test patch with each patch of memory bank
    dist_score, _ = torch.min(distances, dim=1)       # Val and index of the distance scores (minimum values of each row in distances)

    return dist_score