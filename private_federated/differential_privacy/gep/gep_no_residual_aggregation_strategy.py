# import numpy as np
# import torch
# from sklearn.decomposition import PCA
# from private_federated.aggregation_strategies.average_clip_strategy import AverageClipStrategy
# from private_federated.differential_privacy.gep.gep import GEP
# from private_federated.differential_privacy.gep.utils import flatten_tensor
#
#
# class GepNoResidualAggregationStrategy(AverageClipStrategy):
#     def __init__(self, clip_value: float, noise_multiplier: float, gep: GEP):
#         super().__init__(clip_value=clip_value)
#         self._noise_std: float = noise_multiplier * clip_value
#         self._gep: GEP = gep
#
#     def __call__(self, grad_batch: torch.tensor) -> torch.Tensor:
#         flattened_grads = flatten_tensor([grad_batch])
#
#         selected_pca: PCA = self._gep.selected_pca_list[0]
#
#         grad_np: np.ndarray = flattened_grads.cpu().detach().numpy()
#
#         embedding: torch.Tensor = torch.from_numpy(selected_pca.transform(grad_np)).to(grad_batch.device)
#
#         average_clipped_grads = super().__call__(grad_batch=embedding)
#         # Perturbation
#         dp_noise = torch.normal(0, self._noise_std, device=embedding.device) / grad_batch.shape[0]
#
#         perturbed_embedding = average_clipped_grads + dp_noise
#
#         noisy_clipped_embedded_grads = (
#             torch.from_numpy(selected_pca.inverse_transform(perturbed_embedding.cpu()
#                                                             .detach().numpy()))
#             .to(perturbed_embedding.device))
#         return noisy_clipped_embedded_grads
