"""
Elliot Schumacher, Johns Hopkins University
Created 2/1/19
"""
from codebase import losses
import numpy as np
import torch
from codebase import ranker, negative_sampling, prediction
import torch.optim as optim
from codebase import torch_utils
from tensorboardX import SummaryWriter
import os
import gzip
from time import time
import shutil
import logging
from multiprocessing import Process
from codebase import evaluation
from codebase.sheets import Sheets
import pandas
import pprint
from collections import OrderedDict


def zip_models(models_to_zip, new_zip):
    log = logging.getLogger()
    try:
        start_time = time()
        with gzip.open(new_zip, 'wb') as f_out:
            for path in models_to_zip:
                with open(path, 'rb') as f_in:
                    shutil.copyfileobj(f_in, f_out)
                os.remove(path)
        log.info("Saved models {models} as {zip}, elapsed time {time}".format(models=",".join(models_to_zip),
                                                                              zip=new_zip,
                                                                              time=time() - start_time))
    except:
        log.info("Failed to zip {file}".format(file=models_to_zip))


class PairwiseRankingModel(object):
    """
    This class allows for training and evaluating the ranker using a MentionLinks dataset.

    :arg args: the ConfigArgParse object containing program arguments
    :arg mention_links: MentionLinks object containing the dataset.

    Some code borrowed from Spotlight codebase (see https://github.com/maciejkula/spotlight)
    """

    default_arguments = {
        "n_iter": 100,
        "batch_size": 256,
        "eval_batch_size": 4096,
        "learning_rate": 1e-4,
        "l2": 0.0,
        "num_negative_samples": 10,
        "use_cuda": False,
        "loss": "adaptive_hinge",
        "optimizer": "adam",
        "eval_every": 10,
        "save_every": 10,
        "comb_op" : "max",
        "weight_update_every" : 0,
        "eps_finetune" : 2,
        "n_cands" : 200,
        "threshold_num" : 11,
    }

    def __init__(self,
                 args,
                 mention_links,
                 random_state=None):

        self._n_iter = args.n_iter
        self._learning_rate = args.learning_rate
        self._batch_size = args.batch_size
        self._l2 = args.l2
        self._use_cuda = args.use_cuda
        self._mention_representation = mention_links.mention_representations
        self._concept_representation = mention_links.concept_representations
        self.mention_links = mention_links
        self._optimizer_func = args.optimizer
        self._random_state = random_state or np.random.RandomState()
        self._num_negative_samples = args.num_negative_samples

        self.args = args

        self._num_concepts = None
        self._net = None
        self._optimizer = None

        if args.loss == "hinge":
            self._loss_func = losses.hinge_loss
        elif args.loss == "adaptive_hinge":
            self._loss_func = losses.adaptive_hinge_loss


        self.log = logging.getLogger()
        self.summary_writer = SummaryWriter(os.path.join(args.directory, 'tensorboard'))

        self.model_chkpt_dir = args.directory

        self.last_epoch = 0
        self.last_loss = 0
    def __repr__(self):

        if self._net is None:
            net_representation = '[uninitialised]'
        else:
            net_representation = repr(self._net)

        return ('<{}: {}>'
            .format(
            self.__class__.__name__,
            net_representation,
        ))

    @property
    def _initialized(self):
        return self._net is not None

    def _initialize(self, mention_links):
        """
        Initializes the pytorch model and optimizer

        :param mention_links: MentionLinks dataset
        """

        self._num_concepts = mention_links.num_concepts

        if self.args.nearest_neighbor:
            self.log.info("Using nearest neighbor method")
            self._net = torch_utils.gpu(
                torch.nn.DataParallel(
                    ranker.NearestNeighborCached(args=self.args,
                                            concept_embedding_layer=self._concept_representation,
                                            mention_embedding_layer=self._mention_representation,
                                            mention_links=self.mention_links)),
                self._use_cuda
            )
        elif torch.cuda.device_count() > 1 and not self.args.online:
            self.log.info("Using {0} GPUs!".format(torch.cuda.device_count()))

            self._net = torch_utils.gpu(
                torch.nn.DataParallel(
                    ranker.BertAttentionNeuralRankerCached(args=self.args,
                                        concept_embedding_layer=self._concept_representation,
                                        mention_embedding_layer=self._mention_representation,
                                        mention_links=self.mention_links)),
                self._use_cuda
            )
        elif torch.cuda.device_count() > 1 :
            self.log.info("Using {0} GPUs!".format(torch.cuda.device_count()))

            self._net = torch_utils.gpu(
                torch.nn.DataParallel(
                    ranker.BertAttentionNeuralRanker(args=self.args,
                                            concept_embedding_layer=self._concept_representation,
                                            mention_embedding_layer=self._mention_representation,
                                            mention_links=self.mention_links)),
                self._use_cuda
            )
        elif self.args.online:

            self._net = torch_utils.gpu(
                ranker.BertAttentionNeuralRanker(args=self.args,
                                    concept_embedding_layer=self._concept_representation,
                                    mention_embedding_layer=self._mention_representation,
                                    mention_links=self.mention_links),
                self._use_cuda
            )
        else:

            self._net = torch_utils.gpu(
                ranker.BertAttentionNeuralRankerCached(args=self.args,
                                    concept_embedding_layer=self._concept_representation,
                                    mention_embedding_layer=self._mention_representation,
                                    mention_links=self.mention_links),
                self._use_cuda
            )

        if self._optimizer_func == "adam":
            parameters = self._net.parameters()

            self._optimizer = optim.Adam(
                parameters,
                weight_decay=self._l2,
                lr=self._learning_rate
            )
        if self.args.aux_training is not None and not self.args.combine_aux:
            parameters = list(self._net.aux_layer.parameters()) + list(self._net.lower_layers["mention"].parameters())

            self._aux_optimizer = optim.Adam(
                parameters,
                weight_decay=self._l2,
                lr=self._learning_rate
            )

    def fit_bert_attention(self, mention_links):
        epoch_loss = 0.0

        if self._random_state is None:
            random_state = np.random.RandomState()
        shuffle_indices = np.arange(self.mention_links.mention_representations.shape[0])
        self._random_state.shuffle(shuffle_indices)
        mentions = self.mention_links.mention_representations[shuffle_indices, :]


        concept_ids = self.mention_links.concept_ids[shuffle_indices]
        concepts = self.mention_links.concept_representations[concept_ids, :]
        concept_mask = torch_utils.gpu(mention_links.concept_mask[concept_ids, :], gpu=self._use_cuda)
        mention_indexes = torch_utils.gpu(torch.tensor(self.mention_links.mention_indexes[shuffle_indices, :]), gpu=self._use_cuda)
        mention_mask = torch_utils.gpu(self.mention_links.mention_mask[shuffle_indices, :], gpu=self._use_cuda)

        mention_mask_reduced = torch_utils.gpu(self.mention_links.mention_reduced_mask[shuffle_indices, :], gpu=self._use_cuda)


        minibatch_num = 0
        for i in range(0, len(mentions), self._batch_size):

            batch_mention = mentions[i:i + self._batch_size, :]
            batch_concept = concepts[i:i + self._batch_size, :]

            batch_mention_indexes = mention_indexes[i:i + self._batch_size]

            if self.args.use_att_reg:
                positive_prediction,mention_att_penalty, concept_att_penalty = self._net(ids=batch_mention,
                                                concept_ids=batch_concept,
                                                mention_indexes=batch_mention_indexes,
                                                mention_mask=mention_mask[i:i + self._batch_size, :],
                                                concept_mask=concept_mask[i:i + self._batch_size, :],
                                                mention_mask_reduced = mention_mask_reduced[i:i + self._batch_size, :],
                                                use_att_reg=True
                                                )

            else:
                positive_prediction = self._net(ids=batch_mention,
                                                concept_ids=batch_concept,
                                                mention_indexes=batch_mention_indexes,
                                                mention_mask=mention_mask[i:i + self._batch_size, :],
                                                concept_mask=concept_mask[i:i + self._batch_size, :],
                                                mention_mask_reduced = mention_mask_reduced[i:i + self._batch_size, :],
                                                )


            negative_prediction = negative_sampling.get_multiple_negative_predictions_elmo_att(self,
                batch_mention,
                batch_mention_indexes,
                mention_mask[i:i + self._batch_size, :] ,
                mention_mask_reduced[i:i + self._batch_size, :],
                n=self._num_negative_samples)

            loss = self._loss_func(positive_prediction, negative_prediction)
            if self.args.use_att_reg:


                loss += torch.sum(self.args.att_reg_val * mention_att_penalty/batch_mention.shape[0])
                loss += torch.sum(self.args.att_reg_val * concept_att_penalty/batch_concept.shape[0])

            epoch_loss += float(loss.item())

            self._optimizer.zero_grad()

            loss.backward()
            self._optimizer.step()


            minibatch_num += 1



        return epoch_loss, minibatch_num

    def fit_cached(self, mention_links):
        epoch_loss = 0.0
        concept_ids = mention_links.concept_ids.astype(np.int64)

        mentions, concepts = torch_utils.shuffle(mention_links.mention_ids.astype(np.int64),
                                                 concept_ids,
                                                 random_state=self._random_state)

        mention_ids_tensor = torch_utils.gpu(torch.from_numpy(mentions),
                                             self._use_cuda)
        concept_ids_tensor = torch_utils.gpu(torch.from_numpy(concepts),
                                             self._use_cuda)

        for (minibatch_num,
             (batch_mention,
              batch_concept)) in enumerate(torch_utils.minibatch(mention_ids_tensor,
                                                                 concept_ids_tensor,
                                                                 batch_size=self._batch_size)):

            positive_prediction = self._net(ids=batch_mention, concept_ids=batch_concept)

            if self._loss_func == losses.adaptive_hinge_loss:
                negative_prediction = negative_sampling.get_multiple_negative_predictions(self,
                    batch_mention, n=self._num_negative_samples)
            else:
                negative_prediction = negative_sampling.get_negative_prediction(self, batch_mention)

            self._optimizer.zero_grad()
            loss = self._loss_func(positive_prediction, negative_prediction)
            epoch_loss += float(loss.item())
            if self.args.combine_aux:
                aux_loss = self.aux_training(mention_links)
                loss += self.args.aux_lambda * aux_loss
            loss.backward()
            self._optimizer.step()
        return epoch_loss, minibatch_num

    def aux_training(self, mention_links):
        epoch_loss = 0.0
        eng_ids = mention_links.aux_ids.astype(np.int64)

        l2s, engs = torch_utils.shuffle(mention_links.aux_ids.astype(np.int64),
                                                 eng_ids,
                                                 random_state=self._random_state)

        l2_ids_tensor = torch_utils.gpu(torch.from_numpy(l2s),
                                             self._use_cuda)
        eng_ids_tensor = torch_utils.gpu(torch.from_numpy(engs),
                                             self._use_cuda)

        for (minibatch_num,
             (batch_l2,
              batch_eng)) in enumerate(torch_utils.minibatch(l2_ids_tensor,
                                                                 eng_ids_tensor,
                                                                 batch_size=self._batch_size)):

            positive_prediction = self._net(ids=batch_l2, aux=batch_eng)

            if self._loss_func == losses.adaptive_hinge_loss:
                negative_prediction = negative_sampling.get_multiple_negative_predictions_aux(self,
                    batch_l2, n=self._num_negative_samples)
            else:
                negative_prediction = negative_sampling.get_negative_prediction_aux(self, batch_l2)
            if not self.args.combine_aux:
                self._aux_optimizer.zero_grad()
            if self.args.aux_loss == 'sim':
                loss = losses.sim_loss(positive_prediction, negative_prediction)
            else:
                loss = self._loss_func(positive_prediction, negative_prediction)
            if self.args.combine_aux:
                return loss
            if not self.args.combine_aux:

                epoch_loss += float(loss.item())

                loss.backward()
                self._aux_optimizer.step()
        return epoch_loss, minibatch_num
    
    def fit(self, mention_links, test_dict=None):
        """
        Fit the model.

        :arg mention_links: dataset (a mention_links instance)
        :arg test_dict: test dataset will be evaluated every 'eval_every'
        """

        sheet_obj = Sheets()
        if not self._initialized:
            self._initialize(mention_links)

        models_to_zip = []
        zip_num = 0

        eval_file = os.path.join(self.model_chkpt_dir, "results.csv")
        first_epoch = 0
        if self.last_epoch > 0:
            first_epoch = self.last_epoch + 1
            self.log.info("Starting at epoch {0}".format(first_epoch))


        for epoch_num in range(first_epoch, self._n_iter + first_epoch):
            epoch_loss = 0
            if not self.args.nearest_neighbor:
                self._net.train()

                if self.args.aux_training is not None and not self.args.combine_aux:
                    epoch_loss, minibatch_num = self.aux_training(mention_links)
                    epoch_loss /= minibatch_num + 1

                    self.log.info('Aux Epoch {}: loss {}'.format(epoch_num, epoch_loss))

                if self.args.online:
                    epoch_loss, minibatch_num = self.fit_bert_attention(mention_links)
                else:
                    epoch_loss, minibatch_num = self.fit_cached(mention_links)

                epoch_loss /= minibatch_num + 1

                self.log.info('Epoch {}: loss {}'.format(epoch_num, epoch_loss))

                self.last_epoch = epoch_num
                self.last_loss = epoch_loss


            # tensorboard stuff
            self.summary_writer.add_scalar('loss', epoch_loss, epoch_num)

            if self.args.embedding == "bert" and self.args.online and self.args.eps_finetune == (epoch_num+1):
                try:
                    for param in self._net.bert.parameters():
                        param.requires_grad = False
                except:
                    for param in self._net.module.bert.parameters():
                        param.requires_grad = False
                self.log.warning("Frozen bert weights at epoch {0}".format(epoch_num))

            for name, param in self._net.named_parameters():
                self.summary_writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch_num)

            # save checkpoint
            if ((epoch_num + 1) % int(self.args.save_every)) == 0:

                model_path = os.path.join(self.model_chkpt_dir, "checkpoint_{0}.tar".format(epoch_num))
                torch.save({
                    'epoch': epoch_num,
                    'model_state_dict': self._net.state_dict(),
                    'optimizer_state_dict': self._optimizer.state_dict(),
                    'loss': epoch_loss,
                },
                    model_path)
                models_to_zip.append(model_path)
                if len(models_to_zip) == 1:
                    archive_name = os.path.join(self.model_chkpt_dir, "chkpt_archive_{0}.tar.gz".format(epoch_num))
                    p = Process(target=zip_models, args=(models_to_zip, archive_name))
                    p.start()
                    zip_num += 1
                    models_to_zip = []


            #evaluation
            if ((epoch_num + 1) % int(self.args.eval_every)) == 0 and test_dict is not None:
                with torch.no_grad():

                    self.log.info("Evaluating at {0}".format(epoch_num))
                    if self.args.online:
                        predictions = prediction.predict_bert_att(self, mention_links)
                    else:
                        predictions = prediction.predict_faster(self, mention_links, test_dict)

                    output_path = os.path.join(self.model_chkpt_dir, "eval_{0}".format(epoch_num))

                    scores, tac_result_list = evaluation.score(mention_links, predictions, test_dict, outpath=output_path)
                    tac_results = evaluation.tac_script(tac_result_list, epoch_num,
                                                        outpath=self.model_chkpt_dir, args=self.args)
                    scores.update(tac_results)
                    try:
                        for key, val in scores.items():
                            self.summary_writer.add_scalar(key, val, epoch_num)
                    except:
                        pass
                    pp = pprint.PrettyPrinter(indent=4)
                    self.log.info("Epoch:{0}")
                    self.log.info(pp.pformat(scores))


                    scores['epoch'] = epoch_num
                    scores['epoch_loss'] = epoch_loss
                    eval_first = not os.path.exists(eval_file)

                    with open(eval_file, 'a') as eval_csv:
                        dataframe = pandas.DataFrame.from_dict([scores])
                        dataframe.to_csv(eval_csv, header=eval_first, index=False)

                    run = {**vars(self.args), **scores}

                    sheet_obj.update_run(run)



    def eval_saved(self, mention_links, model_path, test_dict):
        if not self._initialized:
            self._initialize(mention_links)

        self.model_chkpt_dir = os.path.dirname(model_path)
        self.load_model(model_path)
        self.log.info("Evaluating model saved at {0}".format(model_path))
        if self.args.embedding == "elmo" and self.args.online:
            predictions = self.predict_elmo_online(mention_links)

        else:
            predictions = self.predict(mention_links)
        output_path = os.path.join(os.path.dirname(model_path), "eval.csv")

        scores = evaluation.score(mention_links, predictions, test_dict, outpath=output_path)
        pprinter = pprint.PrettyPrinter()
        self.log.info("Scores:\n{0}".format(pprinter.pformat(scores)))

    def compare_saved(self, mention_links, model_path, test_dict):
        if not self._initialized:
            self._initialize(mention_links)

        #self.load_model(model_path)
        #self.log.info("Evaluating model saved at {0}".format(model_path))
        self.compare(mention_links, test_dict)

    def load_model(self, path: str):
        """
        Load a previously trained pytorch model
        :param path: A string containing the path name for the model
        """

        to_delete = False
        if "tar.gz" in path:
            new_path = path.replace("tar.gz", "tar")
            with gzip.open(path, 'rb') as f_in:
                with open(new_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            path = new_path
            to_delete = True

        device = 'cpu'
        if self.args.use_cuda:
            device = 'cuda'

        checkpoint = torch.load(path, map_location=device)
        ignore_states = ["mention_embeddings"]
        checkpoint_dict = checkpoint['model_state_dict']
        # Ignore mention_embeddings from previous runs
        if len(ignore_states) > 0:
            new_checkpoint_dict = self._net.state_dict()
            for k, v in checkpoint_dict.items():
                excluded = False
                for ig in ignore_states:
                    if ig in k:
                        excluded = True
                        print("Excluding {0}".format(k))
                if not excluded:
                    new_checkpoint_dict[k] = v
            checkpoint_dict = new_checkpoint_dict
        try:
            self._net.load_state_dict(checkpoint_dict)
            self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except:
            # Handle loading a model to a CPU which was trained on GPU
            new_state_dict = OrderedDict()
            for k, v in checkpoint_dict.items():
                name = k.replace("module.", "")  # remove `module.`
                new_state_dict[name] = v
            self._net.load_state_dict(new_state_dict)
            op_new_state_dict = OrderedDict()
            for k, v in checkpoint['optimizer_state_dict'].items():
                name = k.replace("module.", "")  # remove `module.`
                op_new_state_dict[name] = v
            self._optimizer.load_state_dict(op_new_state_dict)
        self.last_epoch = checkpoint['epoch']
        self.last_loss = checkpoint['loss']
        if to_delete:
            try:
                os.remove(path)
                self.log.info("Deleted {0}".format(path))
            except Exception as e:
                self.log.info("Deleting {0} failed, error:{0}".format(e))

        self.log.info("Loaded model {path}.\nLoss:{loss}, epoch:{epoch}".format(path=path,
                                                                                epoch=self.last_epoch,
                                                                                loss=self.last_loss))
