"""
 from Spotlight codebase (see https://github.com/maciejkula/spotlight)


Classes describing datasets of mention-concept interactions. Instances of these
are returned by dataset-fetching and dataset-processing functions.
"""
import math
import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel, XLMRobertaModel, XLMRobertaTokenizer
from codebase import torch_utils
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from collections import Counter, defaultdict
from data.Objects import *
import os
import pickle
class MentionLinks(object):
    """
    This object encapsulates the storage of representations (embeddings) for text mentions and an ontology of concepts.
    Given a document with span annotations, this object provides reference to labeled concepts during training alongside
    access to a validation set of mentions for use in prediction.
    """

    default_arguments = {
        "embedding": "elmo",
        "emb_layer" : 0,
        "bert_model": "bert-base-multilingual-cased",
        "bert_path": "",
        "bert_path_kb" : "",
        "include_nil": False,
        "test_exclude_nils":False,
        "online": False,
        "test_limit": None,
        "ont_emb" : False,
        "ont_w2v_filename": "",
        "ont_id_mapping": "",
        "ont_name_mapping": "",
        "dnorm_feats" : "",
        "concept_context": "",
        "attention" : False,
        "neg_samps": "",
        "max_neg_samples" : 100,
        "metamap": False,
        "max_seq_length" : 400,
        "max_context_length": 150,

        "context_layer" : -1,
        "include_context" : False,
        "include_typing" : False,
        "aux_pairs" : 100000,

    }

    bert_models = {
        "bert-large-uncased" : 1024,
        "bert-base-uncased" : 768,
        "bert-large-cased" : 1024,
        "bert-base-cased" : 768
    }




    def __init__(self, args, train_entities, ontology_entities, test_entities=None):
        """

        :param comm_dict: a dictionary mapping training document ids to document text and mention annotations.
        :param args: the arguments of the training/prediction run
        :param test_comm_dict: a dictionary mapping validation document ids to document text and mention annotations.
        """
        self.log = logging.getLogger()
        self.args = args

        if args.embedding == "bert":
            self.load_bert_online_att(args, train_entities, ontology_entities, test_entities)

    def get_embeddings_batch(self, concept_chars, model, args, layer, out_dim_size, concept_att, concept_mask, men_max_seq_length=None, context=False, is_mention=False):
        max_len = 0
        padding_id = self.tokenizer.convert_tokens_to_ids(["[PAD]"])
        second_representations = None

        concept_chars = torch_utils.gpu(concept_chars,  args.use_cuda)
        if not args.attention:
            representations = torch_utils.gpu(torch.zeros(size=(len(concept_chars),out_dim_size)),  args.use_cuda)
        else:
            if men_max_seq_length is not None:
                representations = torch_utils.gpu(torch.zeros(size=(len(concept_chars),men_max_seq_length,out_dim_size)),  args.use_cuda)

            else:
                representations = torch_utils.gpu(torch.zeros(size=(len(concept_chars),args.max_seq_length,out_dim_size)),  args.use_cuda)
        if is_mention:
            second_representations = torch_utils.gpu(torch.zeros(size=(len(concept_chars),out_dim_size)),  args.use_cuda)
        concept_mask = torch_utils.gpu(concept_mask,  args.use_cuda)
        for i in range(0, len(concept_chars), args.batch_size):
            upper = min(i+args.batch_size, len(concept_chars))

            with torch.no_grad():
                _, _, mention_embeddings_layers = model(concept_chars[i:upper , :])
                mention_embeddings_all = mention_embeddings_layers[self.args.emb_layer]
                if context:
                    embedding_layer = mention_embeddings_layers[self.args.context_layer]
                    concept_embedding = embedding_layer[:, 0, :]
                    representations[i:upper , :] = concept_embedding

                elif args.comb_op.lower() == "max" and not args.attention:
                    embedding_layer = mention_embeddings_all

                    if is_mention:
                        embedding_layer = mention_embeddings_layers[self.args.context_layer]
                        mention_context = embedding_layer[:, 0, :]
                        second_representations[i:upper, :] = mention_context

                    masked_concept = (embedding_layer + concept_mask[i:upper , :]
                                      .view(embedding_layer.size()[0], embedding_layer.size()[1], 1)
                                      .expand(embedding_layer.size()[0], embedding_layer.size()[1],
                                              embedding_layer.size()[2]))

                    concept_embedding = masked_concept.max(dim=1)[0]

                    representations[i:upper , :] = concept_embedding
                else:
                    if men_max_seq_length is not None:
                        this_concept_mask = concept_mask[i:upper, :]
                        mention_embedding_masked = mention_embeddings_all[
                            concept_mask.unsqueeze(2).expand(this_concept_mask.shape[0], this_concept_mask.shape[1],
                                                             mention_embeddings_all.shape[2])].view(
                            mention_embeddings_all.shape[0], men_max_seq_length,
                            mention_embeddings_all.shape[2])
                        representations[i:upper, :men_max_seq_length, :] = mention_embedding_masked
                    else:
                        representations[i:upper , :args.max_seq_length, :] = mention_embeddings_all[:, :args.max_seq_length, :]


        return representations, second_representations

    def get_embeddings_batch_safe(self, concept_chars, model, args, layer, out_dim_size, concept_att, concept_mask,
                                  men_max_seq_length=None, context=False, is_mention=False):
        representations = None
        second_representations = None
        while representations is None and args.batch_size > 0:
            try:
                representations, second_representations = self.get_embeddings_batch(concept_chars, model, args, layer, out_dim_size, concept_att, concept_mask,
                                     men_max_seq_length, context, is_mention)
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    self.log.warning("Error:{0}".format(str(e)))
                    args.batch_size = math.ceil(args.batch_size / 2.)
                    self.log.info("Reducing batch size to {0}".format(args.batch_size))
        if representations is None:
            self.log.error("Batch size ran out")
            raise Exception("Batch size not found")
        return representations, second_representations

    def tokenize_bert(self, text, longest_seq_context):
        this_context = "[CLS] {0} [SEP]".format(text)

        tokenized_context = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.tokenize(this_context))
        if len(tokenized_context) > self.args.max_seq_length:
            tokenized_context = tokenized_context[:self.args.max_seq_length]
        longest_seq_context = max(len(tokenized_context), longest_seq_context)
        return longest_seq_context, tokenized_context


    def load_concept_embeddings(self, args, bert_model, concept_chars, out_dim_size):
        concept_context_representations = None
        fn_suffix = "{0}_{1}_{2}_{3}_{4}_{5}".format(args.emb_layer, args.comb_op, args.context_layer, args.max_seq_length,
                                                  args.language, args.bert_path.replace("/", ""))
        if len(args.holdout_language) > 0:
            fn_suffix += f"_{args.holdout_language.replace(',', '_')}"
        if args.evaluation:
            fn_suffix += f"_eval"
        if args.bert_path != 'bert-base-multilingual-cased':
            cr_filename_pt = os.path.join(args.data_directory, f"concept_{fn_suffix}.pt")
            cr_filename_pkl = os.path.join(args.data_directory, f"concept_{fn_suffix}.pkl")
        else:
            cr_filename_pt = os.path.join(args.data_directory, "concept_{0}_{1}_{2}_{3}_{4}.pt"
                                          .format(args.emb_layer, args.comb_op, args.context_layer, args.max_seq_length,
                                                  args.language))
            cr_filename_pkl = os.path.join(args.data_directory, "concept_{0}_{1}_{2}_{3}_{4}.pkl"
                                           .format(args.emb_layer, args.comb_op, args.context_layer,
                                                   args.max_seq_length, args.language))
        self.log.info(f"Checking file : {cr_filename_pt}")
        if not os.path.exists(cr_filename_pt):
            self.log.info(f"Creating file : {cr_filename_pt}")

            concept_representations, _ = self.get_embeddings_batch_safe(concept_chars=self.concept_representations,
                                                                        model=bert_model,
                                                                        args=args,
                                                                        layer=args.emb_layer,
                                                                        out_dim_size=768,
                                                                        concept_att=self.concept_att,
                                                                        concept_mask=self.concept_mask)
            self.log.info(f"Concept size : {concept_representations.size()}")
            torch.save(concept_representations, cr_filename_pt)
            with open(cr_filename_pkl, 'wb') as pkl_f:
                pickle.dump(self.cui_to_concept_info, pkl_f)
            self.log.info('Saved concept representations to  {0}'.format(cr_filename_pt))
        else:
            if torch.cuda.is_available():
                cached_concept_representations = torch.load(cr_filename_pt)
            else:
                cached_concept_representations = torch.load(cr_filename_pt, map_location=torch.device('cpu'))

            concept_representations = torch.zeros(size=(len(concept_chars),out_dim_size))

            with open(cr_filename_pkl, 'rb') as pkl_f:
                cached_cui_to_concept_info = pickle.load(pkl_f)
            from_cache = 0
            from_model = 0
            for concept in self.cui_to_concept_info:
                new_i = self.cui_to_concept_info[concept][0]["index"]
                if concept in cached_cui_to_concept_info:
                    from_cache += 1
                    new_i = self.cui_to_concept_info[concept][0]["index"]
                    cached_i = cached_cui_to_concept_info[concept][0]["index"]
                    concept_representations[new_i, :] = cached_concept_representations[cached_i, :]
                else:
                    from_model += 1
                    concept_representations[new_i, :], _ = self.get_embeddings_batch_safe(
                        concept_chars=self.concept_representations[new_i, :].unsqueeze(0),
                        model=bert_model,
                        args=args,
                        layer=args.emb_layer,
                        out_dim_size=768,
                        concept_att=self.concept_att[new_i, :].unsqueeze(0),
                        concept_mask=self.concept_mask[new_i, :].unsqueeze(0))
            self.log.info("Loaded {0} from cache, {1} from model".format(from_cache, from_model))

        if args.include_context:
            if args.bert_path != 'bert-base-multilingual-cased':
                cr_filename_pt = os.path.join(args.data_directory, f"concept_context_{fn_suffix}.pt")
                cr_filename_pkl = os.path.join(args.data_directory, f"concept_context_{fn_suffix}.pkl")
            else:
                cr_filename_pt = os.path.join(args.data_directory, "concept_context_{0}_{1}_{2}_{3}_{4}.pt"
                                           .format(args.emb_layer, args.comb_op, args.context_layer, args.max_seq_length, args.language))
                cr_filename_pkl = os.path.join(args.data_directory, "concept_context_{0}_{1}_{2}_{3}_{4}.pkl"
                                           .format(args.emb_layer, args.comb_op, args.context_layer, args.max_seq_length, args.language))
            if not os.path.exists(cr_filename_pt):
                concept_context_representations, _ = self.get_embeddings_batch_safe(
                    concept_chars=self.concept_context_representations,
                    model=bert_model,
                    args=args,
                    layer=args.emb_layer,
                    out_dim_size=768,
                    concept_att=self.concept_context_att,
                    concept_mask=self.concept_context_mask)
                torch.save(concept_context_representations, cr_filename_pt)
                with open(cr_filename_pkl, 'wb') as pkl_f:
                    pickle.dump(self.cui_to_concept_info, pkl_f)
                self.log.info('Saved concept representations to  {0}'.format(cr_filename_pt))
            else:
                cached_context_concept_representations = torch.load(cr_filename_pt)
                concept_context_representations = torch.zeros(size=(len(concept_chars),out_dim_size))

                with open(cr_filename_pkl, 'rb') as pkl_f:
                    cached_cui_to_concept_info = pickle.load(pkl_f)
                for concept in self.cui_to_concept_info:
                    new_i = self.cui_to_concept_info[concept][0]["index"]
                    if concept in cached_cui_to_concept_info:
                        new_i = self.cui_to_concept_info[concept][0]["index"]
                        cached_i = cached_cui_to_concept_info[concept][0]["index"]
                        concept_context_representations[new_i, :] = cached_context_concept_representations[cached_i, :]
                    else:
                        concept_context_representations[new_i, :] = self.get_embeddings_batch_safe(
                            concept_chars=self.concept_context_representations[new_i, :],
                            model=bert_model,
                            args=args,
                            layer=args.emb_layer,
                            out_dim_size=768,
                            concept_att=self.concept_context_att[new_i, :],
                            concept_mask=self.concept_context_mask[new_i, :])
        return concept_representations, concept_context_representations

    def load_aux_training(self, args, bert_model):
        eng_longest_seq = 0
        l2_longest_seq = 0

        name_pairs_df = pd.read_csv(args.aux_training)
        english_names = []
        l2_names = []

        for i, pair in name_pairs_df.sample(args.aux_pairs,  replace = False).iterrows():
            eng_longest_seq, eng_txt = self.tokenize_bert(pair["ENG"], eng_longest_seq)
            english_names.append(eng_txt)
            l2_longest_seq, l2_txt = self.tokenize_bert(pair["L2"], l2_longest_seq)
            l2_names.append(l2_txt)
            
        padding_id_kb = self.tokenizer_kb.convert_tokens_to_ids(["[PAD]"])

        self.aux_eng_representations = torch.zeros((len(english_names), eng_longest_seq), dtype=torch.long) + padding_id_kb[0]
        self.aux_eng_mask = torch.zeros((self.aux_eng_representations.shape[0], self.aux_eng_representations.shape[1])) + self.mask_false_val
        self.aux_eng_att = torch.zeros((self.aux_eng_representations.shape[0], self.aux_eng_representations.shape[1])) + self.mask_false_val
        self.aux_ids = np.zeros(self.aux_eng_representations.shape[0])

        for i, cp in enumerate(english_names):
            self.aux_eng_representations[i, :len(cp)] = torch.tensor(cp)
            self.aux_eng_mask[i, :len(cp)] = self.mask_true_val
            self.aux_eng_att[i, :len(cp)] = self.mask_true_val
            self.aux_ids[i] = i
        from time import  time
        start = time()
        self.aux_eng_representations, _ = self.get_embeddings_batch_safe(
            concept_chars=self.aux_eng_representations,
            model=bert_model,
            args=args,
            layer=args.emb_layer,
            out_dim_size=768,
            concept_att=self.aux_eng_mask,
            concept_mask=self.aux_eng_att)
        self.log.info(f"Aux eng processed in {time() - start}")

        self.aux_l2_representations = torch.zeros((len(l2_names), eng_longest_seq), dtype=torch.long) + padding_id_kb[
            0]
        self.aux_l2_mask = torch.zeros(
            (self.aux_l2_representations.shape[0], self.aux_l2_representations.shape[1])) + self.mask_false_val
        self.aux_l2_att = torch.zeros(
            (self.aux_l2_representations.shape[0], self.aux_l2_representations.shape[1])) + self.mask_false_val
    
        for i, cp in enumerate(l2_names):
            self.aux_l2_representations[i, :len(cp)] = torch.tensor(cp)
            self.aux_l2_mask[i, :len(cp)] = self.mask_true_val
            self.aux_l2_att[i, :len(cp)] = self.mask_true_val
        start = time()

        self.aux_l2_representations, _ = self.get_embeddings_batch_safe(
            concept_chars=self.aux_l2_representations,
            model=bert_model,
            args=args,
            layer=args.emb_layer,
            out_dim_size=768,
            concept_att=self.aux_l2_mask,
            concept_mask=self.aux_l2_att)
        self.log.info(f"Aux l2 processed in {time() - start}")
        self.log.info(f"Aux reps {self.aux_eng_representations.size()}")

    def mention_types(self, mention_type_list, included_types, mention_mapping_ids = None):
        if mention_mapping_ids is None:
            mention_type_counter = defaultdict(lambda : 0)
            for t_list in mention_type_list:
                for t in t_list:
                    mention_type_counter[t] += 1
            for i in range(len(mention_type_list)):
                type_set = mention_type_list[i].copy()
                for t in mention_type_list[i]:
                    if mention_type_counter[t] < 100:
                        type_set.remove(t)
                    else:
                        included_types.add(t)
                mention_type_list[i] = type_set
            mention_mapping_ids = {}
        mention_type = np.zeros(shape=(len(mention_type_list), len(included_types)))

        for i, t_list in enumerate(mention_type_list):
            for t in t_list:
                if t in mention_mapping_ids:
                    mention_type[i, mention_mapping_ids[t]] = 1
        mention_type = torch.Tensor(mention_type)
        return mention_type, mention_mapping_ids, included_types

    def load_bert_online_att(self, args, train_entities : Mentions, ontology_entities: Ontology, test_entities: Mentions):
        self.only_annotated_concepts = args.only_annotated_concepts

        self.mask_true_val = 1.
        self.mask_false_val = 0.
        if "xlm" in args.bert_path.lower():

            self.tokenizer = XLMRobertaTokenizer.from_pretrained(args.bert_path)

        else:
            # do_lower_case="uncased" in args.bert_model)
            self.tokenizer = BertTokenizer.from_pretrained(args.bert_path)
                                                      #do_lower_case="uncased" in args.bert_model)
        if len(self.args.bert_path_kb) > 0:
            self.tokenizer_kb = BertTokenizer.from_pretrained(args.bert_path_kb)
        else:
            self.tokenizer_kb = self.tokenizer
        self.test_data = test_entities is not None
        self.cui_to_concept_info = {}
        self.id_to_concept_info = {}
        #concept_names = []

        if not args.online:
            self.mask_false_val = float("-inf")
            self.mask_true_val = 0.
            if "xlm" in args.bert_path.lower():
                bert_model = XLMRobertaModel.from_pretrained(args.bert_path,output_hidden_states=True)

            else:
                bert_model = BertModel.from_pretrained(args.bert_path,output_hidden_states=True)
            bert_model = torch_utils.gpu(bert_model, args.use_cuda)
            
        if args.aux_training is not None:
            self.load_aux_training(args, bert_model)
        self.included_concepts = set()
        if self.only_annotated_concepts:
            for te in train_entities.entities:
                self.included_concepts.add(te.kbid)

        concept_parts = []
        longest_seq = 0
        longest_seq_context = 0
        index = 0
        self.concept_type_list = []
        self.concept_popularity = []
        concept_context = []
        no_context = 0

        total_popularity = float(sum(e.popularity for e in ontology_entities.entities.values()))
        for oe in ontology_entities.entities.values():

            if self.only_annotated_concepts and oe.id not in self.included_concepts:
                continue

            elif not self.only_annotated_concepts:
                self.included_concepts.add(oe.id)
            if args.popularity:
                self.concept_popularity.append(oe.popularity / total_popularity)
            concept_map = {"name": oe.name,
                           "concept_id": oe.id,
                           "alternate": False,
                           "index": index
                           }
            self.cui_to_concept_info[oe.id] = [concept_map]
            self.id_to_concept_info[index] = concept_map

            longest_seq, tokenized_name = \
                self.tokenize_bert(oe.name, longest_seq)
            concept_parts.append(tokenized_name)
            if args.include_context:

                context_text = ontology_entities.get_text(oe)
                if context_text.startswith("f_m"):
                    no_context += 1
                longest_seq_context, tokenized_context = \
                    self.tokenize_bert(context_text, longest_seq_context)
                concept_context.append(tokenized_context)

            oe.index = index
            if len(oe.type) > 0:
                self.concept_type_list.append(set(oe.type))
            else:
                self.concept_type_list.append({"None"})

            index += 1
        self.log.info("Percentage with no context : {0}".format(no_context / float(len(concept_parts))))
        if args.popularity:
            self.concept_popularity = torch.Tensor(self.concept_popularity)
            self.log.info(f"Adding popularity {self.concept_popularity}")
        concept_type_counter = defaultdict(lambda : 0)
        for t_list in self.concept_type_list:
            for t in t_list:
                concept_type_counter[t] += 1

        omitted = 0
        included_types = set()
        for i in range(len(self.concept_type_list)):
            type_set = self.concept_type_list[i].copy()
            for t in self.concept_type_list[i]:
                if concept_type_counter[t] < 100:
                    type_set.remove(t)
                    omitted += 1
                else:
                    included_types.add(t)
            self.concept_type_list[i] = type_set
        self.log.info("{0} concept types omitted".format(omitted))

        self.concept_type = np.zeros(shape=(len(self.concept_type_list), len(included_types)))
        concept_mapping_ids = {}
        for i, t_list in enumerate(self.concept_type_list):
            for t in t_list:
                if t not in concept_mapping_ids:
                    concept_mapping_ids[t] = len(concept_mapping_ids)
                self.concept_type[i, concept_mapping_ids[t]] = 1
        self.concept_type = torch.Tensor(self.concept_type)

        padding_id_kb = self.tokenizer_kb.convert_tokens_to_ids(["[PAD]"])
        padding_id = self.tokenizer.convert_tokens_to_ids(["[PAD]"])

        self.concept_representations = torch.zeros((len(concept_parts), longest_seq), dtype=torch.long) + padding_id_kb[0]
        self.concept_mask = torch.zeros((self.concept_representations.shape[0], self.concept_representations.shape[1])) + self.mask_false_val
        self.concept_att = torch.zeros((self.concept_representations.shape[0], self.concept_representations.shape[1])) + self.mask_false_val

        for i, cp in enumerate(concept_parts):
            self.concept_representations[i, :len(cp)] = torch.tensor(cp)
            self.concept_mask[i, :len(cp)] = self.mask_true_val
            self.concept_att[i, :len(cp)] = self.mask_true_val


        if args.include_context:
            self.concept_context_representations = torch.zeros((len(concept_context), longest_seq_context), dtype=torch.long) + \
                                           padding_id_kb[0]
            self.concept_context_mask = torch.zeros(
                (self.concept_context_representations.shape[0], self.concept_context_representations.shape[1])) + self.mask_false_val
            self.concept_context_att = torch.zeros(
                (self.concept_context_representations.shape[0], self.concept_context_representations.shape[1])) + self.mask_false_val

            for i, cp in enumerate(concept_context):
                self.concept_context_representations[i, :len(cp)] = torch.tensor(cp)
                self.concept_context_mask[i, :len(cp)] = self.mask_true_val
                self.concept_context_att[i, :len(cp)] = self.mask_true_val

        if not args.online:
            self.concept_representations, self.concept_context_representations = \
                self.load_concept_embeddings(args, bert_model, self.concept_representations, 768)

        else:
            self.concept_mask = self.concept_mask > self.mask_false_val
            self.concept_att = self.concept_att > self.mask_false_val

        self.num_concepts = len(self.concept_representations)

        self.log.info("Concept reps processed")


        self.num_examples = sum([1 for mention in train_entities.entities if mention.kbid in self.included_concepts])

        self.mention_ids = np.zeros(self.num_examples)
        self.concept_ids = np.zeros(self.num_examples)
        self.mention_type_list = []

        mention_sentences = []

        self.concepts_used = set()
        self.concepts_ids_used = set()

        indx = 0
        self.mention_indexes = []
        men_longest_seq = 0
        self.id_to_mention_info = {}
        entity_binarizer = OneHotEncoder(sparse=False, handle_unknown='ignore')

        from time import  time

        for j, me in enumerate(train_entities.entities):
            if (me.kbid in self.included_concepts or not self.only_annotated_concepts):
                mention_map = {"comm_uuid": me.doc_filename,
                               "mention_uuid": me.id,
                               "index": indx
                               }
                self.id_to_mention_info[indx] = mention_map

                sent_list = self.mention_context(me, args)

                self.mention_indexes.append([1 + token.i - sent_list[0].start for token in me.mention])
                if me.nil:
                    self.concept_ids[indx] = -1
                else:
                    self.concept_ids[indx] = ontology_entities.entities[me.kbid].index
                self.mention_ids[indx] = indx
                self.concepts_used.add(me.kbid)
                self.concepts_ids_used.add(ontology_entities.entities[me.kbid].index)
                sentence = []
                for s in sent_list:
                    for w in s:
                        sentence.append(w.text.strip())
                sentence.insert(0, "[CLS]")
                sentence.append("[SEP]")
                self.mention_type_list.append([me.kb_type])

                mention_sentences.append(sentence)
                me.index = indx
                indx += 1
                if (j+1) % 500 == 0:
                    self.log.info(f"Processed {j} mentions")



        omitted = 0
        included_types = set()
        if args.dataset == 'LDC2019T02':
            self.mention_type = torch.Tensor(entity_binarizer.fit_transform(self.mention_type_list))
        else:
            self.mention_type, mention_mapping_ids, included_types = self.mention_types(self.mention_type_list, included_types)

        length = len(sorted(self.mention_indexes, key=len, reverse=True)[0])
        self.mention_indexes = np.array([xi+[-1]*(length-len(xi)) for xi in self.mention_indexes])
        start = time()
        self.mention_representations, self.mention_mask, self.mention_att, self.mention_reduced_mask = self.bert_tokenize(mention_sentences,
                                                                             self.mention_indexes,
                                                                             self.tokenizer,
                                                                             self.mask_true_val,
                                                                             self.mask_false_val,
                                                                             padding_id,
                                                                           self.mask_true_val,
                                                                           self.mask_false_val)
        self.max_men_length = self.mention_reduced_mask.shape[1]
        if not args.online:
            self.mention_representations, self.mention_context_representations = self.get_embeddings_batch_safe(concept_chars=self.mention_representations,
                                                                     model=bert_model,
                                                                      args=args,
                                                                      layer=args.emb_layer,
                                                                      out_dim_size=768,
                                                                      concept_att=self.mention_att,
                                                                      concept_mask=self.mention_mask,
                                                                     men_max_seq_length=length,
                                                                     is_mention=True)
        end = time()
        self.log.info(f"Mention reps processed in {end-start}")

        if self.test_data:
            test_mention_indexes_list = []
            test_mention_sentences = []
            self.test_num_examples = sum(
                [1 for mention in test_entities.entities if (mention.kbid in self.included_concepts) or
                 (mention.nil and not args.test_exclude_nils)])

            self.test_concept_ids = np.zeros(self.test_num_examples)
            self.test_mention_ids = []
            self.test_mention_ids_pred = np.zeros(self.test_num_examples)
            self.test_id_to_mention_info = {}
            self.test_mention_type_list = []
                                                            
            indx = 0
            for j, me in enumerate(test_entities.entities):
                if me.kbid in self.included_concepts or not self.only_annotated_concepts \
                        or (me.nil and not args.test_exclude_nils):
                    mention_map = {"comm_uuid": me.doc_filename,
                                   "mention_uuid": me.id,
                                   "index": indx
                                   }
                    self.test_id_to_mention_info[indx] = mention_map
                    sent_list = self.mention_context(me, args)
                    test_mention_indexes_list.append([1 + token.i - sent_list[0].start for token in me.mention])
                    self.test_mention_ids.append(me.id)
                    self.test_mention_ids_pred[indx] = indx
                    self.concepts_used.add(me.kbid)

                    sentence = []
                    for s in sent_list:
                        for w in s:
                            sentence.append(w.text.strip())
                    sentence.insert(0, "[CLS]")
                    sentence.append("[SEP]")

                    test_mention_sentences.append(sentence)
                    me.index = indx
                    if me.nil:
                        self.test_concept_ids[indx] = -1
                    else:
                        self.test_concept_ids[indx] = ontology_entities.entities[me.kbid].index
                        self.concepts_ids_used.add(ontology_entities.entities[me.kbid].index)

                    self.test_mention_type_list.append([me.kb_type])
                    indx += 1

                    if args.test_limit is not None and indx == args.test_limit - 1:
                        self.log.warning("Limiting test dataset to {0}".format(args.test_limit))
                        self.test_mention_ids = self.test_mention_ids[:args.test_limit]
                        self.test_concept_ids = self.test_concept_ids[:args.test_limit]
                        self.test_num_examples = args.test_limit
                        break
                    if (j + 1) % 500 == 0:
                        self.log.info(f"Processed {j} test mentions")
            length = len(sorted(test_mention_indexes_list, key=len, reverse=True)[0])
            self.test_mention_indexes = np.array([xi + [-1] * (length - len(xi)) for xi in test_mention_indexes_list])
            if args.dataset == 'LDC2019T02':
                self.test_mention_type = torch.Tensor(entity_binarizer.transform(self.test_mention_type_list))
            else:
                self.test_mention_type, mention_mapping_ids, included_types = \
                    self.mention_types(self.test_mention_type_list, included_types, mention_mapping_ids)
            start = time()
            self.test_mention_representations, self.test_mention_mask, self.test_mention_att, self.test_mention_reduced_mask = self.bert_tokenize(test_mention_sentences,
                                                                                 self.test_mention_indexes,
                                                                                 self.tokenizer,
                                                                                 self.mask_true_val,
                                                                                 self.mask_false_val,
                                                                                 padding_id,
                                                                              self.mask_true_val,
                                                                              self.mask_false_val)

            if not args.online:
                self.test_mention_representations, self.test_mention_context_representations = self.get_embeddings_batch_safe(concept_chars=self.test_mention_representations,
                                                                              model=bert_model,
                                                                              args=args,
                                                                              layer=args.emb_layer,
                                                                              out_dim_size=768,
                                                                              concept_att=self.test_mention_att,
                                                                              concept_mask=self.test_mention_mask,
                                                                                men_max_seq_length=length,
                                                                     is_mention=True)
            end = time()
            self.log.info(f"Test reps processed in {end-start}")

        self.log.info("Loaded bert characters")
        self.log.info("Size of kb reps:{0}".format(self.concept_representations.size()))
        self.log.info("Size of men reps:{0}".format(self.mention_representations.size()))
        self.log.info("Size of men type reps:{0}".format(self.mention_type.size()))
        self.log.info("Size of con type reps:{0}".format(self.concept_type.size()))

        self.log.info("Size of test men reps:{0}".format(self.test_mention_representations.size()))
        self.log.info("Size of test con reps:{0}".format(self.test_mention_context_representations.size()))
        self.log.info("Size of test type reps:{0}".format(self.test_mention_type.size()))

    def mention_context(self, me, args):
        sent = me.mention.sent
        sent_list = [sent]
        sentence_length = len(sent)
        before_sents = sorted(list(s for s in sent.doc.sents if s.start < sent.start), key=lambda x: x.start)
        after_sents = sorted(list(s for s in sent.doc.sents if s.start > sent.start), key=lambda x: x.start)
        for i in range(max(len(before_sents), len(after_sents))):
            added = False
            if i < len(after_sents) and sentence_length < args.max_context_length:
                sent_list.append(after_sents[i])
                sentence_length += len(after_sents[i])
                added = True
            neg_i = -(i + 1)
            if neg_i >= -len(before_sents) and sentence_length < args.max_context_length:
                sent_list.insert(0, before_sents[neg_i])
                sentence_length += len(before_sents[neg_i])
                added = True
            if not added:
                break
        return sent_list
    def bert_tokenize(self, sentences, mention_indexes, tokenizer, mask_true_val, mask_false_val, padding_id, att_true_val, att_false_val):
        mention_parts = []
        longest_seq = 0
        longest_men = 0

        mention_mask_list = []
        mention_att_list = []
        mention_size_list = []
        for sent, mi in zip(sentences, mention_indexes):
            this_sentence  = []
            this_mask = []
            this_att = []
            this_mention_size = 0
            mention_indexes_subword = []
            for i_tok, tok in enumerate(sent):
                this_tok = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(tok))
                if i_tok in mi:
                    this_mask.extend([mask_true_val for x in this_tok])
                    this_att.extend([att_true_val for x in this_tok])
                    this_mention_size += len(this_tok)
                    mention_indexes_subword.extend([j + len(this_sentence) for j, _ in enumerate(this_tok)])
                else:
                    this_mask.extend([mask_false_val for x in this_tok])
                    this_att.extend([att_false_val for x in this_tok])
                this_sentence.extend(this_tok)

            if len(this_sentence) > self.args.max_seq_length:
                min_span = min(mention_indexes_subword)
                max_span = max(mention_indexes_subword)
                t_span_length = max_span - min_span + 1
                padding_len = math.floor((self.args.max_seq_length - t_span_length) / 2)
                lower_bound = max(min_span - padding_len, 0)
                upper_bound = min(max_span + padding_len, len(this_sentence))
                this_sentence = this_sentence[lower_bound:upper_bound]
                this_mask = this_mask[lower_bound:upper_bound]
                this_att = this_att[lower_bound:upper_bound]
            mention_parts.append(this_sentence)
            mention_mask_list.append(this_mask)
            mention_att_list.append(this_att)
            mention_size_list.append(this_mention_size)
            longest_seq = max(longest_seq, len(this_sentence))
            longest_men = max(longest_men, sum(1 for x in this_mask if x == mask_true_val))


        mention_representations = torch.zeros((len(mention_parts), longest_seq + longest_men), dtype=torch.long) + padding_id[0]
        mention_reduced_mask = torch.zeros((len(mention_parts), longest_men)) + mask_false_val

        for i, cp in enumerate(mention_parts):
            mention_representations[i, :len(cp)] = torch.tensor(cp)
            mention_reduced_mask[i, :mention_size_list[i]] = mask_true_val

        mention_mask = torch.zeros((mention_representations.shape[0], mention_representations.shape[1])) + mask_false_val
        mention_att = torch.zeros((mention_representations.shape[0], mention_representations.shape[1])) + att_false_val

        for k in range (len(mention_representations)):
            mention_mask[k, :len(mention_mask_list[k])] = torch.tensor(mention_mask_list[k])
            mention_att[k, :len(mention_att_list[k])] = torch.tensor(mention_att_list[k])
            if self.args.attention:
                upper = mention_mask.shape[1]
                lower = upper - (longest_men -  sum(1 for x in mention_mask_list[k] if x == mask_true_val))
                mention_mask[k, lower:upper] = self.mask_true_val

                upper = mention_att.shape[1]
                lower = upper - (longest_men -  sum(1 for x in mention_att_list[k] if x == mask_true_val))
                mention_att[k, lower:upper] = self.mask_true_val

        if self.args.attention:
            mention_mask = mention_mask > self.mask_false_val
            mention_att = mention_att > self.mask_false_val
            mention_reduced_mask = mention_reduced_mask > self.mask_false_val
        return mention_representations, mention_mask, mention_att, mention_reduced_mask


    def __len__(self):

        return len(self.mention_ids)

