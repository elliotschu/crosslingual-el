"""
Elliot Schumacher, Johns Hopkins University
Created 2/25/20
"""

import torch
from codebase import torch_utils
import logging
def sample_concepts(scorer, num_concepts, shape, random_state=None):
    """
    Randomly sample a number of concepts.

    Parameters
    ----------

    num_concepts: int
        Total number of concepts from which we should sample:
        the maximum value of a sampled concept id will be smaller
        than this.
    shape: int or tuple of ints
        Shape of the sampled array.
    random_state: np.random.RandomState instance, optional
        Random state to use for sampling.

    Returns
    -------

    concepts: np.array of shape [shape]
        Sampled concept ids.
    """

    """if random_state is None:
        random_state = np.random.RandomState()

    concepts = random_state.randint(0, num_concepts, shape, dtype=np.int64)
    """
    concepts = torch.LongTensor(shape).random_(0, num_concepts)

    return concepts


def get_negative_prediction(scorer, mention_ids, concept_ids=None):
    negative_concepts = sample_concepts(scorer,
        scorer._num_concepts,
        len(mention_ids),
        random_state=scorer._random_state)

    negative_var = torch_utils.gpu(negative_concepts, scorer._use_cuda)

    negative_prediction = scorer._net(ids=mention_ids, concept_ids=negative_var)

    return negative_prediction


def get_multiple_negative_predictions(scorer, mention_ids, n=5):
    batch_size = mention_ids.size(0)

    negative_prediction = get_negative_prediction(scorer, mention_ids
                                                        .view(batch_size, 1)
                                                        .expand(batch_size, n)
                                                        .reshape(batch_size * n))

    return negative_prediction.view(n, len(mention_ids))
def get_negative_prediction_aux(scorer, l2_ids, concept_ids=None):
    negative_concepts = sample_concepts(scorer,
        scorer.args.aux_pairs,
        len(l2_ids),
        random_state=scorer._random_state)

    negative_var = torch_utils.gpu(negative_concepts, scorer._use_cuda)

    negative_prediction = scorer._net(ids=l2_ids, aux=negative_var)

    return negative_prediction


def get_multiple_negative_predictions_aux(scorer, l2_ids, n=5):
    batch_size = l2_ids.size(0)

    negative_prediction = get_negative_prediction_aux(scorer, l2_ids
                                                        .view(batch_size, 1)
                                                        .expand(batch_size, n)
                                                        .reshape(batch_size * n))

    return negative_prediction.view(n, len(l2_ids))

def get_multiple_negative_predictions_elmo_att(scorer, mention_ids, batch_mention_index, batch_mention_mask,
                                                batch_mention_reduced_mask, n=5):
    batch_size = mention_ids.size(0)
    if len(mention_ids.size()) == 3:
        reshaped_mentions = mention_ids \
            .view(1, mention_ids.size()[0], mention_ids.size()[1], mention_ids.size()[2]) \
            .expand(n, mention_ids.size()[0], mention_ids.size()[1], mention_ids.size()[2]) \
            .reshape(n * mention_ids.size()[0], mention_ids.size()[1], mention_ids.size()[2])
    else:
        reshaped_mentions = mention_ids \
            .view(1, mention_ids.size()[0], mention_ids.size()[1]) \
            .expand(n, mention_ids.size()[0], mention_ids.size()[1]) \
            .reshape(n * mention_ids.size()[0], mention_ids.size()[1])
    reshaped_indexes = batch_mention_index \
        .view(1, batch_mention_index.shape[0], batch_mention_index.shape[1]) \
        .expand(n, batch_mention_index.shape[0], batch_mention_index.shape[1]) \
        .reshape(n * batch_mention_index.shape[0], batch_mention_index.shape[1])
    reshaped_masks = batch_mention_mask \
        .view(1, batch_mention_mask.shape[0], batch_mention_mask.shape[1]) \
        .expand(n, batch_mention_mask.shape[0], batch_mention_mask.shape[1]) \
        .reshape(n * batch_mention_mask.shape[0], batch_mention_mask.shape[1])
    reshaped_reduced_masks = batch_mention_reduced_mask \
        .view(1, batch_mention_reduced_mask.shape[0], batch_mention_reduced_mask.shape[1]) \
        .expand(n, batch_mention_reduced_mask.shape[0], batch_mention_reduced_mask.shape[1]) \
        .reshape(n * batch_mention_reduced_mask.shape[0], batch_mention_reduced_mask.shape[1])

    negative_prediction = _get_negative_prediction_elmo_att(scorer, reshaped_mentions, reshaped_indexes, reshaped_masks,
                                                                 reshaped_reduced_masks)

    return negative_prediction.view(n, mention_ids.size()[0])


def _get_negative_prediction_elmo_att(scorer, mention_ids, batch_mention_index, reshaped_masks, reshaped_reduced_masks):
    # concepts = self._random_state.randint(0, self._num_concepts, len(mention_ids), dtype=np.int64)
    concepts = torch.LongTensor(len(mention_ids)).random_(0, scorer._num_concepts)

    this_concept_mask = torch_utils.gpu(scorer.mention_links.concept_mask[concepts], gpu=scorer._use_cuda)

    negative_var = torch_utils.gpu(scorer.mention_links.concept_representations[concepts], gpu=scorer._use_cuda)

    negative_prediction = scorer._net(ids=mention_ids, concept_ids=negative_var,
                                    mention_indexes=batch_mention_index,
                                    concept_mask=this_concept_mask,
                                    mention_mask=reshaped_masks,
                                    mention_mask_reduced=reshaped_reduced_masks)

    return negative_prediction


