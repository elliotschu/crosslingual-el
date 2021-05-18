"""
Elliot Schumacher, Johns Hopkins University
Created 2/12/19

"""
from time import time
import torch.nn as nn
import torch
from collections import OrderedDict
import numpy as np
from codebase import torch_utils
import logging
from transformers import BertTokenizer, BertModel
from codebase.self_attention import StructuredSelfAttention
import os


class BertNeuralRanker(nn.Module):
    """

    :arg args: ConfigArgParse object containing program arguments
    :arg mention_embedding_layer: If using cached embeddings, embeddings for mentions
    :arg concept_embedding_layer: If using cached embeddings, embeddings for concepts
    :arg mention_links: A MentionLinks dataset
    :arg transform: The transformation function for n-1 layers
    :arg final_transform: The transformation function for the nth layer
    """

    default_arguments = {
        "num_hidden_layers": 4,
        "hidden_layer_size": "512",
        "dropout_prob": 0.2,
        "freeze_emb_concept": True,
        "freeze_emb_mention": True,
        "elmo_options_file": "",
        "elmo_weight_file": "",
        "finetune_elmo": False,
        "elmo_mix" : "",
        "weighted_only" : False
    }

    def __init__(self,
                 args,
                 mention_embedding_layer,
                 concept_embedding_layer,
                 mention_links,
                 transform = nn.ReLU,
                 final_transform=nn.Tanh):

        super(BertNeuralRanker, self).__init__()
        self.output_size = 1
        self.args = args
        input_size = mention_embedding_layer.shape[1] + concept_embedding_layer.shape[1]

        self.log = logging.getLogger()
        self.dnorm_features = False

        if args.embedding == "bert" and not args.online:
            self.mention_embeddings = nn.Embedding.from_pretrained(mention_embedding_layer,freeze=args.freeze_emb_mention)

            self.concept_embeddings = nn.Embedding.from_pretrained(concept_embedding_layer,freeze=args.freeze_emb_concept)

        elif args.embedding == "bert" and args.online:
            bert_model = BertModel.from_pretrained(args.bert_path)
            self.bert = torch_utils.gpu(bert_model, args.use_cuda)

            #self.bert.eval()

            input_size = 768 * 2
            self.emb_size = 768


        self.transform = transform
        self.final_transform = final_transform
        self.num_hidden_layers = args.num_hidden_layers
        self.layers = OrderedDict()

        if not self.args.weighted_only:

            if "," in args.hidden_layer_size:
                self.hidden_layer_size = [int(x) for x in args.hidden_layer_size.split(",")]
                if len(self.hidden_layer_size) != self.num_hidden_layers + 1:
                    raise Exception("Wrong hidden layer size specification")
            else:
                self.hidden_layer_size = [int(args.hidden_layer_size) for _ in range(self.num_hidden_layers + 1)]

            self.layers['linear_input'] = nn.Linear(input_size,self.hidden_layer_size[0])
            self.layers['transform_input'] = self.transform()
            if args.dropout_prob > 0.0:
                self.layers['dropout_input'] = nn.Dropout(args.dropout_prob)

            for i in range(0, args.num_hidden_layers):
                self.layers['linear_h{i}'.format(i=i)] = nn.Linear(self.hidden_layer_size[i],
                                                                   self.hidden_layer_size[i+1])
                self.layers['transform_h{i}'.format(i=i)] = self.transform()
                if args.dropout_prob > 0.0:
                    self.layers['dropout_h{i}'.format(i=i)] = nn.Dropout(args.dropout_prob)

            self.layers['linear_output'] = nn.Linear(self.hidden_layer_size[-1], self.output_size)
            self.layers['transform_output'] = self.final_transform()

            self.sequential_net = nn.Sequential(self.layers)
            self.log.info("Sequential net:{0}".format(self.sequential_net))

        else:
            self.weight1 = nn.Parameter(torch.Tensor(self.mention_embeddings.weight.size()[1],
                                                     self.concept_embeddings.weight.size()[1]))
            nn.init.xavier_uniform_(self.weight1)
        self.log.info("Module:{0}".format(self))


    def forward(self, ids, concept_ids=None, mention_indexes=None, mention_mask=None, concept_mask=None,
                cached_emb=False, emb_only=False, mention_att=None, concept_att=None):
        """
        Runs a forward pass over the model.  This has several different usages depending on the situation, documented below.


        :return loss of model
        """

        if emb_only:
            #This only returns the underlying embedding of the bert or elmo model.  This is used for caching in prediction.
            embedding_layers, embedding_pool = self.bert(ids)
            return embedding_pool
        elif self.args.online and cached_emb:
            if concept_mask is not None: #ids will contain mention embedding
                mention_embedding = ids.squeeze()
                concept_chars = concept_ids


                concept_embedding_layers, concept_embedding_pool = self.bert(concept_chars[0,:].view(1,concept_chars.shape[1])
                                                                             ,attention_mask=concept_att[0,:].view(1,concept_chars.shape[1]))


                if self.args.comb_op == "max":
                    concept_embedding_all = concept_embedding_layers[self.args.emb_layer]\
                        .expand(concept_chars.shape[0],concept_embedding_layers.shape[1])

                    masked_concept = (concept_embedding_all + concept_mask[0,:]
                                      .view(concept_embedding_all.size()[0], concept_embedding_all.size()[1], 1)
                                      .expand(concept_embedding_all.size()[0], concept_embedding_all.size()[1],
                                              concept_embedding_all.size()[2]))

                    concept_embedding = masked_concept.max(dim=1)[0]
                    concept_embedding = concept_embedding.view(1, concept_embedding.shape[1]) \
                        .expand(len(mention_embedding), concept_embedding.shape[1])
                elif self.args.comb_op == "cls" and self.args.embedding == "bert":
                    concept_embedding = concept_embedding_pool\
                        .expand(concept_chars.shape[0],concept_embedding_pool.shape[1])
            else:
                mention_embedding = ids.squeeze()
                concept_embedding = concept_ids.squeeze()

        elif self.args.online and not cached_emb:
            if concept_ids is None:
                mention_chars = ids[0]
                concept_chars = ids[1]
                mention_indexes = ids[2]
            else:
                mention_chars = torch_utils.gpu(ids, gpu=self.args.use_cuda)
                concept_chars = torch_utils.gpu(concept_ids, gpu=self.args.use_cuda)
                mention_indexes = mention_indexes

            concept_embedding_layers, concept_embedding_pool = self.bert(concept_chars, attention_mask=concept_att)
            concept_embedding_all = concept_embedding_layers[self.args.emb_layer]


            mention_embeddings_layers, mention_embedding_pool = self.bert(mention_chars, attention_mask=mention_att)
            mention_embeddings_all = mention_embeddings_layers[self.args.emb_layer]

            if self.args.comb_op == "max":
                masked_concept = (concept_embedding_all + concept_mask
                               .view(concept_embedding_all.size()[0], concept_embedding_all.size()[1], 1)
                               .expand(concept_embedding_all.size()[0], concept_embedding_all.size()[1],
                                       concept_embedding_all.size()[2]))

                concept_embedding = masked_concept.max(dim=1)[0]

                masked_mens = (mention_embeddings_all + mention_mask
                               .view(mention_embeddings_all.size()[0], mention_embeddings_all.size()[1], 1)
                               .expand(mention_embeddings_all.size()[0], mention_embeddings_all.size()[1],
                                       mention_embeddings_all.size()[2]))

                mention_embedding = masked_mens.max(dim=1)[0]
            elif self.args.comb_op == "cls" and self.args.embedding == "bert":
                concept_embedding = concept_embedding_pool
                mention_embedding = mention_embedding_pool
        else:
            # This is used when there are cached embeddings provided in a lookup table.
            if concept_ids is None:

                mention_embedding = self.mention_embeddings(ids[0])
                concept_embedding = self.concept_embeddings(ids[1])
                concept_ids = ids[1]
                ids = ids[0]
            else:
                mention_embedding = self.mention_embeddings(ids)
                concept_embedding = self.concept_embeddings(concept_ids)

        input_rep = torch.cat([mention_embedding, concept_embedding], 1)


        out = self.sequential_net(input_rep)

        return out

class NearestNeighborCached(nn.Module):
    def __init__(self,
                 args,
                 mention_embedding_layer,
                 concept_embedding_layer,
                 mention_links,
                 transform = nn.ReLU,
                 final_transform=nn.Tanh):

        super(NearestNeighborCached, self).__init__()
        self.output_size = 1
        self.args = args

        self.log = logging.getLogger()

        self.emb_size = 768

        self.transform = transform
        self.final_transform = final_transform
        self.layers = OrderedDict()

        self.mention_embeddings = torch_utils.gpu(
            nn.Embedding.from_pretrained(mention_embedding_layer, freeze=args.freeze_emb_mention), args.use_cuda)

        self.concept_embeddings = torch_utils.gpu(
            nn.Embedding.from_pretrained(concept_embedding_layer, freeze=args.freeze_emb_concept), args.use_cuda)
        self.attention = False

        self.similarity = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, ids, concept_ids=None, cached_emb=False):

        if cached_emb:
            mention_embedding = ids
            concept_embedding = self.concept_embeddings(concept_ids)

        elif concept_ids is None:

            mention_embedding = self.mention_embeddings(ids[0])
            concept_embedding = self.concept_embeddings(ids[1])

        else:
            mention_embedding = self.mention_embeddings(ids)
            concept_embedding = self.concept_embeddings(concept_ids)

        return self.similarity(mention_embedding, concept_embedding)


class BertAttentionNeuralRankerCached(nn.Module):
    """

    :arg args: ConfigArgParse object containing program arguments
    :arg mention_embedding_layer: If using cached embeddings, embeddings for mentions
    :arg concept_embedding_layer: If using cached embeddings, embeddings for concepts
    :arg mention_links: A MentionLinks dataset
    :arg transform: The transformation function for n-1 layers
    :arg final_transform: The transformation function for the nth layer
    """

    default_arguments = {
        "hidden_layer_size": "512",
        "dropout_prob": 0.2,
        "freeze_emb_concept": True,
        "freeze_emb_mention": True,
        "weighted_only" : False,
        "bert_config" : "bert_config.json",
        "att_heads": 1,
        "att_dim": 256,
        "use_att_reg": False,
        "att_reg_val": 0.0001,
        "context_hidden_layer_size" : "",
        "mention_hidden_layer_size": "",
        "type_hidden_layer_size": "",
        "aux_loss": "hidden",
    }

    def __init__(self,
                 args,
                 mention_embedding_layer,
                 concept_embedding_layer,
                 mention_links,
                 transform = nn.ReLU,
                 final_transform=nn.Tanh):

        super(BertAttentionNeuralRankerCached, self).__init__()
        self.output_size = 1
        self.args = args
        input_size = mention_embedding_layer.shape[1] + concept_embedding_layer.shape[1]

        self.log = logging.getLogger()


        input_size = 768 * 2
        self.emb_size = 768

        self.transform = transform
        self.final_transform = final_transform
        self.layers = OrderedDict()

        if self.args.attention:
            self.mention_embeddings = torch_utils.gpu(mention_embedding_layer, args.use_cuda)

            self.concept_embeddings = torch_utils.gpu(concept_embedding_layer, args.use_cuda)

            self.mention_attention = torch_utils.gpu(StructuredSelfAttention(lstm_hid_dim=self.emb_size,
                                                                             d_a=args.att_dim,
                                                                             r=args.att_heads,
                                                                             max_len=mention_links.max_men_length,
                                                                             use_gpu=args.use_cuda), args.use_cuda)
            self.concept_attention = torch_utils.gpu(StructuredSelfAttention(lstm_hid_dim=self.emb_size,
                                                                             d_a=args.att_dim,
                                                                             r=args.att_heads,
                                                                             max_len=concept_embedding_layer.shape[1],
                                                                             use_gpu=args.use_cuda), args.use_cuda)
            self.attention = True
        else:
            self.mention_embeddings = torch_utils.gpu(
                nn.Embedding.from_pretrained(mention_embedding_layer, freeze=args.freeze_emb_mention), args.use_cuda)

            self.concept_embeddings = torch_utils.gpu(
                nn.Embedding.from_pretrained(concept_embedding_layer, freeze=args.freeze_emb_concept), args.use_cuda)
            self.attention = False

        self.hidden_layer_size = [int(x) for x in args.hidden_layer_size.split(",")]

        if args.include_context or args.include_typing:
            if args.include_typing:
                self.mention_type_emb = torch_utils.gpu(
                    nn.Embedding.from_pretrained(mention_links.mention_type, freeze=True),
                    args.use_cuda)

                self.concept_type_emb = torch_utils.gpu(
                    nn.Embedding.from_pretrained(mention_links.concept_type, freeze=True),
                    args.use_cuda)
            if args.include_context:
                self.mention_context_emb = torch_utils.gpu(
                    nn.Embedding.from_pretrained(mention_links.mention_context_representations, freeze=True),
                    args.use_cuda)

                self.concept_context_emb = torch_utils.gpu(
                    nn.Embedding.from_pretrained(mention_links.concept_context_representations, freeze=True),
                    args.use_cuda)
            self.lower_layers, input_upper = self.define_layers(args, mention_links)
            input_size = input_upper
        else:
            self.lower_layers = None

        if args.popularity:
            self.concept_popularity_embeddings = torch_utils.gpu(
                nn.Embedding.from_pretrained(torch.unsqueeze(mention_links.concept_popularity, 1), freeze=True), args.use_cuda)
            input_size += 1

        if args.aux_training is not None:

            if args.aux_loss == "hidden":
                aux_input = self.mention_hidden_layer_size[-1]
                self.aux_layer = nn.Linear(aux_input, 1)
                self.aux_layer = torch_utils.gpu(self.aux_layer, args.use_cuda)
                self.log.info(self.aux_layer)
                self.aux_transform = self.final_transform()

            self.aux_eng_embeddings = torch_utils.gpu(
                nn.Embedding.from_pretrained(mention_links.aux_eng_representations, freeze=args.freeze_emb_mention), args.use_cuda)

            self.aux_l2_embeddings = torch_utils.gpu(
                nn.Embedding.from_pretrained(mention_links.aux_l2_representations, freeze=args.freeze_emb_concept), args.use_cuda)


        self.layers['linear_input'] = nn.Linear(input_size,self.hidden_layer_size[0])
        self.layers['transform_input'] = self.transform()
        if args.dropout_prob > 0.0:
            self.layers['dropout_input'] = nn.Dropout(args.dropout_prob)

        for i in range(0, len(self.hidden_layer_size)-1):
            self.layers['linear_h{i}'.format(i=i)] = nn.Linear(self.hidden_layer_size[i],
                                                               self.hidden_layer_size[i+1])
            self.layers['transform_h{i}'.format(i=i)] = self.transform()
            if args.dropout_prob > 0.0:
                self.layers['dropout_h{i}'.format(i=i)] = nn.Dropout(args.dropout_prob)

        self.layers['linear_output'] = nn.Linear(self.hidden_layer_size[-1], self.output_size)
        self.layers['transform_output'] = self.final_transform()

        self.sequential_net = nn.Sequential(self.layers)

        self.log.info(self.sequential_net)
        try:
            for lay in self.lower_layers:
                self.log.info(lay)
                self.log.info(self.lower_layers[lay])
        except:
            pass


    def define_layers(self, args, mention_links):
        lower_layers = {}
        mention_layers = OrderedDict()
        self.mention_hidden_layer_size = [int(x) for x in args.mention_hidden_layer_size.split(",")]
        mention_layers['mention_linear_input'] = nn.Linear(mention_links.mention_representations.size()[1] +
                                                           mention_links.concept_representations.size()[1],
                                                        self.mention_hidden_layer_size[0])
        mention_layers['mention_transform_input'] = self.transform()
        if args.dropout_prob > 0.0:
            mention_layers['mention_dropout_input'] = nn.Dropout(args.dropout_prob)

        for i in range(0, len(self.mention_hidden_layer_size)-1):
            mention_layers['mention_linear_h{i}'.format(i=i)] = nn.Linear(self.mention_hidden_layer_size[i],
                                                                       self.mention_hidden_layer_size[i + 1])
            mention_layers['mention_transform_h{i}'.format(i=i)] = self.transform()
            if args.dropout_prob > 0.0:
                mention_layers['mention_dropout_h{i}'.format(i=i)] = nn.Dropout(args.dropout_prob)
        input_upper = self.mention_hidden_layer_size[-1]
        lower_layers["mention"] = torch_utils.gpu(nn.Sequential(mention_layers), args.use_cuda)
        if args.include_context:
            context_layers = OrderedDict()
            self.context_hidden_layer_size = [int(x) for x in args.context_hidden_layer_size.split(",")]
            context_layers['context_linear_input'] = nn.Linear(mention_links.mention_context_representations.size()[1] +
                                                               mention_links.concept_context_representations.size()[1],
                                                            self.context_hidden_layer_size[0])
            context_layers['context_transform_input'] = self.transform()
            if args.dropout_prob > 0.0:
                context_layers['context_dropout_input'] = nn.Dropout(args.dropout_prob)

            for i in range(0, len(self.context_hidden_layer_size)-1):
                context_layers['context_linear_h{i}'.format(i=i)] = nn.Linear(self.context_hidden_layer_size[i],
                                                                           self.context_hidden_layer_size[i + 1])
                context_layers['context_transform_h{i}'.format(i=i)] = self.transform()
                if args.dropout_prob > 0.0:
                    context_layers['context_dropout_h{i}'.format(i=i)] = nn.Dropout(args.dropout_prob)
            input_upper += self.context_hidden_layer_size[-1]
            lower_layers["context"] = torch_utils.gpu(nn.Sequential(context_layers), args.use_cuda)

        if args.include_typing:
            type_layers = OrderedDict()
            self.type_hidden_layer_size = [int(x) for x in args.type_hidden_layer_size.split(",")]
            type_layers['type_linear_input'] = nn.Linear(mention_links.mention_type.size()[1] +
                                                         mention_links.concept_type.size()[1],
                                                            self.type_hidden_layer_size[0])
            type_layers['type_transform_input'] = self.transform()
            if args.dropout_prob > 0.0:
                type_layers['type_dropout_input'] = nn.Dropout(args.dropout_prob)

            for i in range(0, len(self.type_hidden_layer_size)-1):
                type_layers['type_linear_h{i}'.format(i=i)] = nn.Linear(self.type_hidden_layer_size[i],
                                                                           self.type_hidden_layer_size[i + 1])
                type_layers['type_transform_h{i}'.format(i=i)] = self.transform()
                if args.dropout_prob > 0.0:
                    type_layers['type_dropout_h{i}'.format(i=i)] = nn.Dropout(args.dropout_prob)

            input_upper += self.type_hidden_layer_size[-1]

            lower_layers["type"] = torch_utils.gpu(nn.Sequential(type_layers), args.use_cuda)

        return lower_layers, input_upper


    def forward(self, ids, concept_ids=None, mention_indexes=None, mention_mask=None, concept_mask=None,
                    cached_emb=False, emb_only=False, mention_att=None, concept_att=None, use_att_reg=False,
                mention_mask_reduced=None, mention_type_emb = None, mention_context_emb=None, aux=None):
            """
            Runs a forward pass over the model.  This has several different usages depending on the situation, documented below.


            :return loss of model
            """
            if aux is None:

                # This is used when there are cached embeddings provided in a lookup table.
                if self.attention:
                    if cached_emb:
                        mention_embedding = ids
                        concept_embedding = self.concept_embeddings[concept_ids, :, :]
                    else:
                        mention_embedding = self.mention_embeddings[ids, :, :]
                        concept_embedding = self.concept_embeddings[concept_ids, :, :]

                    if use_att_reg:
                        mention_embedding, mention_att_penalty = self.mention_attention(mention_embedding, use_reg=True,
                                                                                        mask=mention_mask_reduced)
                        concept_embedding, concept_att_penalty = self.concept_attention(concept_embedding, use_reg=True,
                                                                                        mask=concept_mask)
                    else:
                        mention_embedding = self.mention_attention(mention_embedding, mask=mention_mask_reduced)
                        concept_embedding = self.concept_attention(concept_embedding, mask=concept_mask)
                else:
                    if cached_emb:
                        mention_embedding = ids
                        concept_embedding = self.concept_embeddings(concept_ids)

                    elif concept_ids is None:

                        mention_embedding = self.mention_embeddings(ids[0])
                        concept_embedding = self.concept_embeddings(ids[1])
                        concept_ids = ids[1]
                        ids = ids[0]
                    else:
                        mention_embedding = self.mention_embeddings(ids)
                        concept_embedding = self.concept_embeddings(concept_ids)
                if self.lower_layers:
                    mention_rep = torch.cat([mention_embedding, concept_embedding], 1)
                    input_rep = self.lower_layers["mention"](mention_rep)

                    if self.args.include_typing:
                        if mention_type_emb is None:
                            mention_type_emb = self.mention_type_emb(ids)
                        mention_rep = torch.cat([mention_type_emb,  self.concept_type_emb(concept_ids)], 1)
                        type_intermediate = self.lower_layers["type"](mention_rep)
                        input_rep = torch.cat([input_rep,  type_intermediate], 1)
                    if self.args.include_context:
                        if mention_context_emb is None:
                            mention_context_emb = self.mention_context_emb(ids)
                        mention_rep = torch.cat([mention_context_emb,  self.concept_context_emb(concept_ids)], 1)
                        type_intermediate = self.lower_layers["context"](mention_rep)
                        input_rep = torch.cat([input_rep,  type_intermediate], 1)
                    if self.args.popularity:
                        popularity_rep = self.concept_popularity_embeddings(concept_ids)
                        input_rep = torch.cat([input_rep, popularity_rep], 1)
                    out = self.sequential_net(input_rep)
                else:
                    input_rep = torch.cat([mention_embedding, concept_embedding], 1)
                    if self.args.popularity:
                        popularity_rep = self.concept_popularity_embeddings(concept_ids)
                        input_rep = torch.cat([input_rep, popularity_rep], 1)

                    out = self.sequential_net(input_rep)

            else:
                l2_embedding = self.aux_l2_embeddings(ids)
                eng_embeddings = self.aux_eng_embeddings(aux)

                mention_rep = torch.cat([l2_embedding, eng_embeddings], 1)

                input_rep = self.lower_layers["mention"](mention_rep)
                if self.args.aux_loss == "hidden":
                    out = self.aux_layer(input_rep)
                    out = self.aux_transform(out)
                else:
                    out = input_rep
            return out

class BertAttentionNeuralRanker(nn.Module):
    """

    :arg args: ConfigArgParse object containing program arguments
    :arg mention_embedding_layer: If using cached embeddings, embeddings for mentions
    :arg concept_embedding_layer: If using cached embeddings, embeddings for concepts
    :arg mention_links: A MentionLinks dataset
    :arg transform: The transformation function for n-1 layers
    :arg final_transform: The transformation function for the nth layer
    """

    default_arguments = {
        "num_hidden_layers": 4,
        "hidden_layer_size": "512",
        "dropout_prob": 0.2,
        "freeze_emb_concept": True,
        "freeze_emb_mention": True,
        "weighted_only" : False,
        "bert_config" : "bert_config.json",
        "att_heads": 1,
        "att_dim": 256,
        "use_att_reg": False,
        "att_reg_val": 0.0001
    }

    def __init__(self,
                 args,
                 mention_embedding_layer,
                 concept_embedding_layer,
                 mention_links,
                 transform = nn.ReLU,
                 final_transform=nn.Tanh):

        super(BertAttentionNeuralRanker, self).__init__()
        self.output_size = 1
        self.args = args
        input_size = mention_embedding_layer.shape[1] + concept_embedding_layer.shape[1]

        self.log = logging.getLogger()

        try:
            bert_model = BertModel.from_pretrained(args.bert_path, output_hidden_states=True)
        except:
            bert_model = BertModel.from_pretrained(os.path.join(args.bert_path, 'bert_config.json'), output_hidden_states=True)

        self.bert = torch_utils.gpu(bert_model, args.use_cuda)

        if len(self.args.bert_path_kb) > 0:
            bert_model_kb = BertModel.from_pretrained(args.bert_path_kb, output_hidden_states=True)
            self.bert_kb =  torch_utils.gpu(bert_model_kb, args.use_cuda)
            self.log.info("Using separate bert kb model: {0}".format(args.bert_path_kb))
        else:
            self.bert_kb = self.bert

        #self.bert.eval()

        input_size = 768 * 2
        self.emb_size = 768


        self.transform = transform
        self.final_transform = final_transform
        self.num_hidden_layers = args.num_hidden_layers
        self.layers = OrderedDict()


        self.mention_attention = torch_utils.gpu(StructuredSelfAttention(lstm_hid_dim=self.emb_size,
                                                                         d_a=args.att_dim,
                                                                         r=args.att_heads,
                                                                         max_len=mention_links.max_men_length,
                                                                         use_gpu=args.use_cuda), args.use_cuda)
        self.concept_attention = torch_utils.gpu(StructuredSelfAttention(lstm_hid_dim=self.emb_size,
                                                                         d_a=args.att_dim,
                                                                         r=args.att_heads,
                                                                         max_len=concept_embedding_layer.shape[1],
                                                                         use_gpu=args.use_cuda), args.use_cuda)
        self.attention = True


        if "," in args.hidden_layer_size:
            self.hidden_layer_size = [int(x) for x in args.hidden_layer_size.split(",")]
            if len(self.hidden_layer_size) != self.num_hidden_layers + 1:
                raise Exception("Wrong hidden layer size specification")
        else:
            self.hidden_layer_size = [int(args.hidden_layer_size) for _ in range(self.num_hidden_layers + 1)]

        self.layers['linear_input'] = nn.Linear(input_size,self.hidden_layer_size[0])
        self.layers['transform_input'] = self.transform()
        if args.dropout_prob > 0.0:
            self.layers['dropout_input'] = nn.Dropout(args.dropout_prob)

        for i in range(0, args.num_hidden_layers):
            self.layers['linear_h{i}'.format(i=i)] = nn.Linear(self.hidden_layer_size[i],
                                                               self.hidden_layer_size[i+1])
            self.layers['transform_h{i}'.format(i=i)] = self.transform()
            if args.dropout_prob > 0.0:
                self.layers['dropout_h{i}'.format(i=i)] = nn.Dropout(args.dropout_prob)

        self.layers['linear_output'] = nn.Linear(self.hidden_layer_size[-1], self.output_size)
        self.layers['transform_output'] = self.final_transform()

        self.sequential_net = nn.Sequential(self.layers)
        #self.log.info("Sequential net:{0}".format(self.sequential_net))


        #self.log.info("Module:{0}".format(self))

    def forward(self, ids, concept_ids=None, mention_indexes=None, mention_mask=None, concept_mask=None,
                cached_emb=False, emb_only=False, mention_att=None, concept_att=None, use_att_reg=False, mention_mask_reduced=None):
        """
        Runs a forward pass over the model.  This has several different usages depending on the situation, documented below.


        :return loss of model
        """

        if emb_only:
            #This only returns the underlying embedding of the bert or elmo model.  This is used for caching in prediction.
            #mention_embeddings_layers, mention_embedding_pool = self.bert(ids)

            _, _, mention_embeddings_layers = self.bert(ids)
            mention_embeddings_all = mention_embeddings_layers[self.args.emb_layer]

            mention_embedding_masked = mention_embeddings_all[
                mention_mask.unsqueeze(2).expand(mention_mask.shape[0], mention_mask.shape[1],
                                                        mention_embeddings_all.shape[2])].view(
                mention_embeddings_all.shape[0], mention_mask_reduced.shape[1],
                mention_embeddings_all.shape[2])
            # masked_mens = (mention_embeddings_all + mention_mask
            #                .view(mention_embeddings_all.size()[0], mention_embeddings_all.size()[1], 1)
            #                .expand(mention_embeddings_all.size()[0], mention_embeddings_all.size()[1],
            #                        mention_embeddings_all.size()[2]))
            mention_embedding = self.mention_attention(mention_embedding_masked, mask=mention_mask_reduced)

            return mention_embedding
        elif self.args.online and cached_emb:
            if concept_mask is not None: #ids will contain mention embedding
                mention_embedding = ids.squeeze()
                concept_chars = concept_ids


                #concept_embedding_layers, concept_embedding_pool = self.bert(concept_chars[0,:].view(1,concept_chars.shape[1]))

                _, _, concept_embedding_layers = self.bert_kb(concept_chars[0,:].view(1,concept_chars.shape[1]))

                concept_embedding_all = concept_embedding_layers[self.args.emb_layer]
                                    #\.expand(concept_chars.shape[0],concept_embedding_layers[self.args.emb_layer].shape[1])

                concept_embedding = self.concept_attention(concept_embedding_all, mask=concept_mask[0, :].unsqueeze(0))
                concept_embedding = concept_embedding.expand(len(concept_chars), concept_embedding.shape[1])

            else:
                mention_embedding = ids.squeeze()
                concept_embedding = concept_ids.squeeze()

        elif self.args.online and not cached_emb:
            if concept_ids is None:
                mention_chars = ids[0]
                concept_chars = ids[1]
                mention_indexes = ids[2]
            else:
                mention_chars = torch_utils.gpu(ids, gpu=self.args.use_cuda)
                concept_chars = torch_utils.gpu(concept_ids, gpu=self.args.use_cuda)
                mention_indexes = mention_indexes

            _, _, concept_embedding_layers = self.bert_kb(concept_chars)
            concept_embedding_all = concept_embedding_layers[self.args.emb_layer]

            _, _, mention_embeddings_layers = self.bert(mention_chars)
            mention_embeddings_all = mention_embeddings_layers[self.args.emb_layer]

            mention_embedding_masked = mention_embeddings_all[
                mention_mask.unsqueeze(2).expand(mention_mask.shape[0], mention_mask.shape[1],
                                                        mention_embeddings_all.shape[2])].view(
                mention_embeddings_all.shape[0], mention_mask_reduced.shape[1],
                mention_embeddings_all.shape[2])

            if use_att_reg:
                mention_embedding, mention_att_penalty = self.mention_attention(mention_embedding_masked, use_reg=True,
                                                                                mask=mention_mask_reduced)
                concept_embedding, concept_att_penalty = self.concept_attention(concept_embedding_all, use_reg=True,
                                                                                mask=concept_mask)
            else:
                mention_embedding = self.mention_attention(mention_embedding_masked, mask=mention_mask_reduced)
                concept_embedding = self.concept_attention(concept_embedding_all, mask=concept_mask)

        else:
            # This is used when there are cached embeddings provided in a lookup table.
            if concept_ids is None:

                mention_embedding = self.mention_embeddings(ids[0])
                concept_embedding = self.concept_embeddings(ids[1])
                concept_ids = ids[1]
                ids = ids[0]
            else:
                mention_embedding = self.mention_embeddings(ids)
                concept_embedding = self.concept_embeddings(concept_ids)

        input_rep = torch.cat([mention_embedding, concept_embedding], 1)


        out = self.sequential_net(input_rep)

        return out