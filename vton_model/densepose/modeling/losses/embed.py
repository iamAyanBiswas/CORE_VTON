# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# pyre-unsafe

from typing import Any, Dict, List
import torch
from torch import nn
from torch.nn import functional as F

from vton_model.detectron2.config import CfgNode
from vton_model.detectron2.structures import Instances

from vton_model.densepose.data.meshes.catalog import MeshCatalog
from vton_model.densepose.modeling.cse.utils import normalize_embeddings, squared_euclidean_distance_matrix

from .embed_utils import PackedCseAnnotations
from .utils import BilinearInterpolationHelper


class EmbeddingLoss:
    """
    Computes losses for estimated embeddings given annotated vertices.
    Instances in a minibatch that correspond to the same mesh are grouped
    together. For each group, loss is computed as cross-entropy for
    unnormalized scores given ground truth mesh vertex ids.
    Scores are based on squared distances between estimated vertex embeddings
    and mesh vertex embeddings.
    """

    def __init__(self, cfg: CfgNode):
        """
        Initialize embedding loss from config
        """
        self.embdist_gauss_sigma = cfg.MODEL.ROI_DENSEPOSE_HEAD.CSE.EMBEDDING_DIST_GAUSS_SIGMA

    def __call__(
        self,
        proposals_with_gt: List[Instances],
        densepose_predictor_outputs: Any,
        packed_annotations: PackedCseAnnotations,
        interpolator: BilinearInterpolationHelper,
        embedder: nn.Module,
    ) -> Dict[int, torch.Tensor]:
        """
        Produces losses for estimated embeddings given annotated vertices.
        Embeddings for all the vertices of a mesh are computed by the embedder.
        Embeddings for observed pixels are estimated by a predictor.
        Losses are computed as cross-entropy for squared distances between
        observed vertex embeddings and all mesh vertex embeddings given
        ground truth vertex IDs.

        Args:
            proposals_with_gt (list of Instances): detections with associated
                ground truth data; each item corresponds to instances detected
                on 1 image; the number of items corresponds to the number of
                images in a batch
            densepose_predictor_outputs: an object of a dataclass that contains predictor
                outputs with estimated values; assumed to have the following attributes:
                * embedding - embedding estimates, tensor of shape [N, D, S, S], where
                  N = number of instances (= sum N_i, where N_i is the number of
                      instances on image i)
                  D = embedding space dimensionality (MODEL.ROI_DENSEPOSE_HEAD.CSE.EMBED_SIZE)
                  S = output size (width and height)
            packed_annotations (PackedCseAnnotations): contains various data useful
                for loss computation, each data is packed into a single tensor
            interpolator (BilinearInterpolationHelper): bilinear interpolation helper
            embedder (nn.Module): module that computes vertex embeddings for different meshes
        Return:
            dict(int -> tensor): losses for different mesh IDs
        """
        losses = {}
        for mesh_id_tensor in packed_annotations.vertex_mesh_ids_gt.unique():
            mesh_id = mesh_id_tensor.item()
            mesh_name = MeshCatalog.get_mesh_name(mesh_id)
            # valid points are those that fall into estimated bbox
            # and correspond to the current mesh
            j_valid = interpolator.j_valid * (  # pyre-ignore[16]
                packed_annotations.vertex_mesh_ids_gt == mesh_id
            )
            if not torch.any(j_valid):
                continue
            # extract estimated embeddings for valid points
            # -> tensor [J, D]
            vertex_embeddings_i = normalize_embeddings(
                interpolator.extract_at_points(
                    densepose_predictor_outputs.embedding,
                    slice_fine_segm=slice(None),
                    w_ylo_xlo=interpolator.w_ylo_xlo[:, None],  # pyre-ignore[16]
                    w_ylo_xhi=interpolator.w_ylo_xhi[:, None],  # pyre-ignore[16]
                    w_yhi_xlo=interpolator.w_yhi_xlo[:, None],  # pyre-ignore[16]
                    w_yhi_xhi=interpolator.w_yhi_xhi[:, None],  # pyre-ignore[16]
                )[j_valid, :]
            )
            # extract vertex ids for valid points
            # -> tensor [J]
            vertex_indices_i = packed_annotations.vertex_ids_gt[j_valid]
            # embeddings for all mesh vertices
            # -> tensor [K, D]
            mesh_vertex_embeddings = embedder(mesh_name)
            # unnormalized scores for valid points
            # -> tensor [J, K]
            scores = squared_euclidean_distance_matrix(
                vertex_embeddings_i, mesh_vertex_embeddings
            ) / (-self.embdist_gauss_sigma)
            losses[mesh_name] = F.cross_entropy(scores, vertex_indices_i, ignore_index=-1)

        for mesh_name in embedder.mesh_names:
            if mesh_name not in losses:
                losses[mesh_name] = self.fake_value(
                    densepose_predictor_outputs, embedder, mesh_name
                )
        return losses

    def fake_values(self, densepose_predictor_outputs: Any, embedder: nn.Module):
        losses = {}
        for mesh_name in embedder.mesh_names:
            losses[mesh_name] = self.fake_value(densepose_predictor_outputs, embedder, mesh_name)
        return losses

    def fake_value(self, densepose_predictor_outputs: Any, embedder: nn.Module, mesh_name: str):
        return densepose_predictor_outputs.embedding.sum() * 0 + embedder(mesh_name).sum() * 0
