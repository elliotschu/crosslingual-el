"""
Elliot Schumacher, Johns Hopkins University
Created 2/12/19
"""
import torch

def hinge_loss(positive_predictions, negative_predictions, mask=None, epsilon=1.0):
    """
    Hinge pairwise loss function. (From spotlight https://github.com/maciejkula/spotlight)

    Parameters
    ----------

    positive_predictions: tensor
        Tensor containing predictions for known positive items.
    negative_predictions: tensor
        Tensor containing predictions for sampled negative items.
    mask: tensor, optional
        A binary tensor used to zero the loss from some entries
        of the loss tensor.

    Returns
    -------

    loss, float
        The mean value of the loss function.
    """

    loss = torch.clamp(negative_predictions -
                       positive_predictions +
                       epsilon, 0.0)

    if mask is not None:
        mask = mask.float()
        loss = loss * mask
        return loss.sum() / mask.sum()

    return loss.mean()

def sim_loss(positive_predictions, negative_predictions, mask=None, epsilon=1.0):
    """
    Hinge pairwise loss function. (From spotlight https://github.com/maciejkula/spotlight)

    Parameters
    ----------

    positive_predictions: tensor
        Tensor containing predictions for known positive items.
    negative_predictions: tensor
        Tensor containing predictions for sampled negative items.
    mask: tensor, optional
        A binary tensor used to zero the loss from some entries
        of the loss tensor.

    Returns
    -------

    loss, float
        The mean value of the loss function.
    """
    sim = torch.nn.CosineSimilarity()
    loss = sim(positive_predictions, negative_predictions)

    if mask is not None:
        mask = mask.float()
        loss = loss * mask
        return loss.sum() / mask.sum()

    return loss.mean()

def adaptive_hinge_loss(positive_predictions, negative_predictions, mask=None):
    """
    Adaptive hinge pairwise loss function. Takes a set of predictions
    for implicitly negative items, and selects those that are highest,
    thus sampling those negatives that are closes to violating the
    ranking implicit in the pattern of user interactions.

    Approximates the idea of weighted approximate-rank pairwise loss
    introduced in [2]_

    Parameters
    ----------

    positive_predictions: tensor
        Tensor containing predictions for known positive items.
    negative_predictions: tensor
        Iterable of tensors containing predictions for sampled negative items.
        More tensors increase the likelihood of finding ranking-violating
        pairs, but risk overfitting.
    mask: tensor, optional
        A binary tensor used to zero the loss from some entries
        of the loss tensor.

    Returns
    -------

    loss, float
        The mean value of the loss function.

    References
    ----------

    .. [2] Weston, Jason, Samy Bengio, and Nicolas Usunier. "Wsabie:
       Scaling up to large vocabulary image annotation." IJCAI.
       Vol. 11. 2011.
    """

    highest_negative_predictions, _ = torch.max(negative_predictions, 0)

    return hinge_loss(positive_predictions.squeeze(), highest_negative_predictions.squeeze(), mask=mask)


def main():
    pass


if __name__ == "__main__":
    main()