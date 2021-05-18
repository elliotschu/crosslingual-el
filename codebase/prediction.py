"""
Elliot Schumacher, Johns Hopkins University
Created 2/25/20
"""

from time import time
import numpy as np
import torch
from codebase import torch_utils
def predict_bert_att(scorer, mention_links):
    """
    The prediction module for online bert.  To manage GPU memory, n mention representations (set by eval_batch_size)
    are cached, and then as a batch scored against each concept (one by one).

    :param mention_links: A mention links objects (needs to contain test data)
    :return: a numpy matrix of size [num test mentions] x [num concepts], with each row representing the scores for
    a given mention
    """
    scorer._net.eval()
    if not scorer._initialized:
        scorer._initialize(mention_links)

    # scorer._check_input(mention_links, None, allow_concepts_none=True)
    with torch.no_grad():  # , torch.autograd.profiler.profile(use_cuda=scorer.args.use_cuda) as prof:

        mention_ids = torch_utils.gpu(mention_links.test_mention_representations, scorer.args.use_cuda)
        mention_indxs = torch_utils.gpu(torch.tensor(mention_links.test_mention_indexes), scorer.args.use_cuda)

        # gold_concept_ids = mention_links.test_concept_ids.astype(np.int64)

        results = np.zeros((mention_ids.shape[0], scorer._num_concepts))
        test_mention_mask = torch_utils.gpu(scorer.mention_links.test_mention_mask, gpu=scorer._use_cuda)
        test_mention_reduced_mask = torch_utils.gpu(scorer.mention_links.test_mention_reduced_mask, gpu=scorer._use_cuda)

        start = time()

        try:
            emb_size = scorer._net.emb_size
        except:
            emb_size = scorer._net.module.emb_size
        num_gpus = 1
        if torch.cuda.device_count() > 1:
            num_gpus = torch.cuda.device_count()

        mention_embedding = torch.zeros(len(mention_ids), emb_size)  # this should remain on the cpu!
        for i in range(0, len(mention_ids), scorer.args.eval_batch_size):
            k = min(i + scorer.args.eval_batch_size, len(mention_ids))
            mention_elmo = scorer._net(ids=mention_ids[i:k, :],
                                     emb_only=True,
                                     mention_indexes=mention_indxs[i:k, :],
                                     mention_mask=test_mention_mask[i:k, :],
                                     mention_mask_reduced=test_mention_reduced_mask[i:k, :])
            mention_embedding[i:k, :] = torch_utils.cpu(mention_elmo).detach()

        scorer.log.info("Processed embeddings :{0}".format(time() - start))
        del test_mention_mask
        del mention_indxs
        del mention_ids

        gpu_mention_embeddings = torch_utils.gpu(mention_embedding, gpu=scorer._use_cuda)
        concept_ids = mention_links.concept_representations
        concept_ids = torch_utils.gpu(concept_ids, scorer.args.use_cuda).squeeze()
        # concept_att = torch_utils.gpu(scorer.mention_links.concept_att,  gpu=scorer._use_cuda)
        gpu_concept_mask = torch_utils.gpu(mention_links.concept_mask, gpu=scorer._use_cuda)
        emb_start = time()
        for j in range(0, scorer._num_concepts):

            these_concept_ids = concept_ids[j, :].view(1, concept_ids.shape[1]) \
                .expand(gpu_mention_embeddings.shape[0], concept_ids.shape[1])
            these_concept_mask = gpu_concept_mask[j, :].view(1, mention_links.concept_mask.shape[1]) \
                .expand(gpu_mention_embeddings.shape[0], mention_links.concept_mask.shape[1])
            """

            these_concept_att = concept_att[j, :].view(1, concept_att.shape[1])\
                                                              .expand(gpu_mention_embeddings.shape[0], concept_att.shape[1])"""
            intermed = scorer._net(ids=gpu_mention_embeddings,
                                 concept_ids=these_concept_ids,
                                 concept_mask=these_concept_mask,
                                 cached_emb=True).flatten()
            results[:, j] = torch_utils.cpu(intermed).detach().numpy().transpose()

            if (j + 1) % 10000 == 0:
                # torch.cuda.synchronize()
                scorer.log.info("Processed {0} concepts : {1}".format(j + 1, time() - emb_start))

    # torch.cuda.synchronize()
    # scorer.log.info(prof)
    scorer.log.info("Total eval time={0}".format(time() - start))

    return results

def predict(scorer, mention_links):
    """
    The batch prediction function for cached embedding.  Alternatives are provided for BERT/ELMo in-model embeddings
    :param mention_links: A mention links objects (needs to contain test data)
    :return: a numpy matrix of size [num test mentions] x [num concepts], with each row representing the scores for
    a given mention
    """
    scorer._net.eval()
    if not scorer._initialized:
        scorer._initialize(mention_links)

    mention_ids = np.array(mention_links.test_mention_ids_pred).astype(np.int64)

    concept_ids = np.arange(scorer._num_concepts, dtype=np.int64)
    concept_ids = torch.from_numpy(concept_ids.reshape(-1, 1).astype(np.int64))

    concept_ids = torch_utils.gpu(concept_ids, scorer.args.use_cuda).squeeze()

    mention_ids = torch.from_numpy(mention_ids.reshape(-1, 1).astype(np.int64))
    results = np.zeros((mention_ids.shape[0], scorer._num_concepts), dtype=np.float16)
    for i in range(mention_ids.shape[0]):
        if scorer.args.attention:
            mention_rep = mention_links.test_mention_representations[mention_ids[i], :, :]\
                .expand((concept_ids.size()[0],
                         mention_links.test_mention_representations[mention_ids[i]].shape[1],
                         mention_links.test_mention_representations[mention_ids[i]].shape[2]))
        else:
            mention_rep = mention_links.test_mention_representations[mention_ids[i], :]\
                .expand((concept_ids.size()[0], mention_links.test_mention_representations[mention_ids[i]].shape[1]))
        if scorer.args.include_context and scorer.args.include_typing:
            mention_type_rep = torch_utils.gpu(mention_links.test_mention_type[mention_ids[i], :], scorer.args.use_cuda)\
                .expand((concept_ids.size()[0], mention_links.test_mention_type[mention_ids[i]].shape[1]))
            mention_context_rep = torch_utils.gpu(mention_links.test_mention_context_representations[mention_ids[i], :],
                                               scorer.args.use_cuda) \
                .expand((concept_ids.size()[0], mention_links.test_mention_context_representations[mention_ids[i]].shape[1]))
            results[i, :] = torch_utils.cpu(scorer._net(mention_rep, concept_ids, cached_emb=True,
                                                        mention_type_emb = mention_type_rep,
                                                        mention_context_emb=mention_context_rep).flatten()).detach().numpy()
        elif scorer.args.include_typing:
            mention_type_rep = torch_utils.gpu(mention_links.test_mention_type[mention_ids[i], :], scorer.args.use_cuda)\
                .expand((concept_ids.size()[0], mention_links.test_mention_type[mention_ids[i]].shape[1]))
            results[i, :] = torch_utils.cpu(scorer._net(mention_rep, concept_ids, cached_emb=True,
                                                    mention_type_emb = mention_type_rep).flatten()).detach().numpy()
        elif scorer.args.include_context:
            mention_context_rep = torch_utils.gpu(mention_links.test_mention_context_representations[mention_ids[i], :],
                                               scorer.args.use_cuda) \
                .expand((concept_ids.size()[0], mention_links.test_mention_context_representations[mention_ids[i]].shape[1]))
            results[i, :] = torch_utils.cpu(scorer._net(mention_rep, concept_ids, cached_emb=True,
                                                        mention_context_emb=mention_context_rep).flatten()).detach().numpy()
        else:
            results[i, :] = torch_utils.cpu(scorer._net(mention_rep, concept_ids, cached_emb=True,).flatten()).detach().numpy()

    return results

def predict_faster(scorer, mention_links, test_dict):
    """
    The batch prediction function for cached embedding.  Alternatives are provided for BERT/ELMo in-model embeddings
    :param mention_links: A mention links objects (needs to contain test data)
    :return: a numpy matrix of size [num test mentions] x [num concepts], with each row representing the scores for
    a given mention
    """
    scorer._net.eval()
    if not scorer._initialized:
        scorer._initialize(mention_links)

    mention_ids = np.array(mention_links.test_mention_ids_pred).astype(np.int64)

    #concept_ids = np.arange(scorer._num_concepts, dtype=np.int64)
    #concept_ids = torch.from_numpy(concept_ids.reshape(-1, 1).astype(np.int64))

    #concept_ids = torch_utils.gpu(concept_ids, scorer.args.use_cuda).squeeze()

    mention_ids = torch.from_numpy(mention_ids.reshape(-1, 1).astype(np.int64))
    results = np.zeros((mention_ids.shape[0], scorer.args.n_cands), dtype=np.float16)
    for i, mention_id in enumerate(mention_links.test_mention_ids):
        mention = [mention for mention in test_dict.entities if mention.id == mention_id][0]
        candidate_index_list = []
        for cand_kbid in mention.candidate_kb.keys():
            if cand_kbid in mention_links.cui_to_concept_info:
                candidate_index_list.append([x["index"] for x in mention_links.cui_to_concept_info[cand_kbid]][0])
            else:
                print("----")
                print(cand_kbid)
                print(mention)
                print(i)
                print(mention.id)
                print(str(mention))
        concept_ids = np.asarray(candidate_index_list)
        concept_ids = torch.from_numpy(concept_ids.reshape(-1, 1).astype(np.int64))

        concept_ids = torch_utils.gpu(concept_ids, scorer.args.use_cuda).squeeze()
        if len(concept_ids.size()) > 0:
            mention_rep = mention_links.test_mention_representations[mention_ids[i], :]\
                .expand((concept_ids.size()[0], mention_links.test_mention_representations[mention_ids[i]].shape[1]))

            if scorer.args.include_context and scorer.args.include_typing:
                mention_type_rep = torch_utils.gpu(mention_links.test_mention_type[mention_ids[i], :], scorer.args.use_cuda)\
                    .expand((concept_ids.size()[0], mention_links.test_mention_type[mention_ids[i]].shape[1]))
                mention_context_rep = torch_utils.gpu(mention_links.test_mention_context_representations[mention_ids[i], :],
                                                   scorer.args.use_cuda) \
                    .expand((concept_ids.size()[0], mention_links.test_mention_context_representations[mention_ids[i]].shape[1]))
                results[i, :len(candidate_index_list)] = torch_utils.cpu(scorer._net(mention_rep, concept_ids, cached_emb=True,
                                                            mention_type_emb = mention_type_rep,
                                                            mention_context_emb=mention_context_rep).flatten()).detach().numpy()
            elif scorer.args.include_typing:
                mention_type_rep = torch_utils.gpu(mention_links.test_mention_type[mention_ids[i], :], scorer.args.use_cuda)\
                    .expand((concept_ids.size()[0], mention_links.test_mention_type[mention_ids[i]].shape[1]))
                results[i, :len(candidate_index_list)] = torch_utils.cpu(scorer._net(mention_rep, concept_ids, cached_emb=True,
                                                        mention_type_emb = mention_type_rep).flatten()).detach().numpy()
            elif scorer.args.include_context:
                mention_context_rep = torch_utils.gpu(mention_links.test_mention_context_representations[mention_ids[i], :],
                                                   scorer.args.use_cuda) \
                    .expand((concept_ids.size()[0], mention_links.test_mention_context_representations[mention_ids[i]].shape[1]))
                results[i, :len(candidate_index_list)] = torch_utils.cpu(scorer._net(mention_rep, concept_ids, cached_emb=True,
                                                            mention_context_emb=mention_context_rep).flatten()).detach().numpy()
            else:
                results[i, :len(candidate_index_list)] = torch_utils.cpu(scorer._net(mention_rep, concept_ids, cached_emb=True,)
                                                                         .flatten()).detach().numpy()

    return results
