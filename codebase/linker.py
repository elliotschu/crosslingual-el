"""
Elliot Schumacher, Johns Hopkins University
Created 2/1/19
"""
import torch, sys, time, os, subprocess, logging, configargparse, socket
import traceback
from codebase import ranker
from codebase import scoring
from codebase import mention_links
from codebase import torch_utils
from codebase import evaluation
from codebase.sheets import Sheets
from data import LDC2019T02, Wiki
import shutil
import random
import glob

log = logging.getLogger()


def model(args):

    if args.dataset == 'LDC2019T02':
        if args.generate:
            LDC2019T02.generate_all(args)
            quit()
        elif args.language.upper() == 'ALL':
            train_entities, holdout_entities, ontology = LDC2019T02.load_data_split_all(args)

        else:
            train_entities, holdout_entities, ontology = LDC2019T02.load_data_split(args)
    elif args.dataset == 'wiki':
        if args.language.upper() == 'ALL':
            train_entities, holdout_entities, ontology = Wiki.load_data_split_all(args)
        else:
            train_entities, holdout_entities, ontology = Wiki.load_data_split(args)
    else:
        raise Exception("No dataset {0} ".format(args.dataset))
    links = mention_links.MentionLinks(args, train_entities, ontology, holdout_entities)



    model = scoring.PairwiseRankingModel(args, links)

    #model.compare_saved(mention_links=links, model_path=args.model_path, test_dict=test_dict)

    if args.model_path or args.keep_training:
        if not model._initialized:
            model._initialize(links)
        if args.keep_training:
            try:
                all_files = glob.glob(os.path.join(args.directory, 'chkpt_archive_*.tar.gz'))
                latest_model = max(all_files, key=os.path.getctime)
                log.info("Loading model {0}".format(latest_model))

            except:
                log.error("Cannot find any checkpoints in directory :{0}".format(args.directory))

            model.load_model(latest_model)

        else:
            model.load_model(args.model_path)
    model.fit(links, holdout_entities)

    """
    model.load_model(args.model_path, links)
    predictions = model.predict(links)
    output_path = os.path.join(model.model_chkpt_dir, "eval_final.csv")
    scores = evaluation.score(mention_links, predictions, test_dict, outpath=output_path)
    """


def save_code(args):
    """
    This function zips the current state of the codebase as a replication backup.
    :param args: ConfigArgParse program arguments
    """
    try:
        current_file = sys.argv[0]
        pathname = os.path.abspath(os.path.dirname(current_file))
        outzip = os.path.join(args.directory, "codebase")
        shutil.make_archive(base_name=outzip,
                            format='zip',
                            root_dir=pathname)
        log.info("Saving codebase at {0}".format(outzip))


        outzip = os.path.join(args.directory, "data")
        shutil.make_archive(base_name=outzip,
                            format='zip',
                            root_dir=pathname)
        log.info("Saving data at {0}".format(outzip))

        result = subprocess.run(['git',  'show', '--oneline',  '-s'], stdout=subprocess.PIPE)
        log.info("Git version:\t{0}".format(result.stdout.decode('utf-8')))
    except Exception as e:
        log.error("Error saving code to zip")
        log.error(e)

def parameter_tune(args):
    logging.getLogger().info("Generating parameters for training")
    params = {
        "context_hidden_layer_size": ["768", "512", "256", "512,256"],
        "mention_hidden_layer_size": ["768", "512", "256", "512,256"],
        "type_hidden_layer_size": ["128", "64", "32", "16"],
        "hidden_layer_size": ["512,256", "256,128", "128,64", "1024,512", "512", "256", "512,128"],
        "dropout_prob": [0.1, 0.2, 0.5],
        "learning_rate": [1e-5, 5e-4, 1e-4, 5e-3, 1e-3],
    }

    for par in params:
        this_par = random.choice(params[par])
        setattr(args, par, this_par)
        logging.info("Parameter : {0}, {1}".format(par, this_par))

    return args


if __name__ == "__main__":

    config_file = "../config.ini"
    p = configargparse.ArgParser()
    p.add('-c', '--my-config', is_config_file=True, help='config file path')

    p.add('--train', help='concrete train file')
    p.add('--dev', help='concrete dev file')
    p.add('--test', help='concrete test file')
    p.add_argument('--exclude_nonwiki', dest='exclude_nonwiki', action='store_true')
    p.set_defaults(exclude_nonwiki=False)
    p.add('--dataset', default='LDC2019T02')
    p.add('--root', help='root directory for storing results',default='./results')
    p.add_argument('--only_annotated_concepts', dest='only_annotated_concepts', action='store_true')
    p.set_defaults(only_annotated_concepts=False)
    p.add('--mention_embeddings', help='directory for mention embeddings')
    p.add('--concept_embeddings', help='directory for concept embeddings')
    p.add('--model_path', help='pre-trained model path', default=None)
    p.add('--lexicon', help='path to lexicon', default=None)
    p.add('--data_directory', help='')
    p.add('--language', help='')
    p.add('--holdout_language', help='', default="")
    p.add('--training_perc', help='', default=1.0)
    p.add('--job_num', default="")
    p.add('--max_kb', default=None)
    p.add('--training_ids', default=None)

    p.add('--heldout_max_kb', default=None)
    p.add('--n_triage', default=10, type=int)
    p.add("--popularity_type", default="train")
    p.add('--excluded_types', default="")
    p.add('--lex_min', default = 0.6, type=float)
    p.add_argument('--evaluation', dest='evaluation', action='store_true')
    p.set_defaults(evaluation=False)
    p.add_argument('--generate', dest='generate', action='store_true')
    p.set_defaults(generate=False)
    p.add_argument('--exclude_test_entities', dest='exclude_test_entities', action='store_true')
    p.set_defaults(exclude_test_entities=False)
    p.add_argument('--popularity', dest='popularity', action='store_true')
    p.set_defaults(popularity=False)
    p.add_argument('--nearest_neighbor', dest='nearest_neighbor', action='store_true')
    p.set_defaults(nearest_neighbor=False)
    p.add_argument("--oracle", dest="oracle", action="store_true")
    p.set_defaults(oracle=False)

    p.add("--match_train_lang", default=None, type=str)


    p.add_argument("--combine_aux", dest="combine_aux", action="store_true")
    p.set_defaults(combine_aux=False)

    p.add('--training_ds', default="random")
    p.add('--aux_training', default=None)
    p.add('--aux_lambda', default=0.5, type=float)

    p.add_argument('--reload_cands', dest='reload_cands', action='store_true')
    p.set_defaults(reload_cands=False)

    p.add('--fbt_path', help='')
    p.add('--keep_training', default=False)

    #p.add('--gpus', help='gpus to use', default="0,1")

    p.add('--timestamp', default=None)
    p.add('--directory')

    p.add_argument('--parameter_tune', dest='parameter_tune', action='store_true')
    p.set_defaults(parameter_tune=False)

    # Adds arguments got ranker, mention_links, and scoring classes.
    for arg, val in scoring.PairwiseRankingModel.default_arguments.items():
        if type(val) is not bool:
            p.add("--{0}".format(arg), default=val, type=type(val))
        else:
            p.add_argument('--{0}'.format(arg), dest=arg, action='store_true')
            p.add_argument('--{0}_false'.format(arg), dest=arg, action='store_false')
            p.set_defaults(**{arg:val})

    for arg, val in ranker.BertAttentionNeuralRankerCached.default_arguments.items():
        if type(val) is not bool:
            p.add("--{0}".format(arg), default=val, type=type(val))
        else:
            p.add_argument('--{0}'.format(arg), dest=arg, action='store_true')
            p.add_argument('--{0}_false'.format(arg), dest=arg, action='store_false')
            p.set_defaults(**{arg:val})
    for arg, val in mention_links.MentionLinks.default_arguments.items():
        if type(val) is not bool:
            p.add("--{0}".format(arg), default=val, type=type(val))
        else:
            p.add_argument('--{0}'.format(arg), dest=arg, action='store_true')
            p.add_argument('--{0}_false'.format(arg), dest=arg, action='store_false')
            p.set_defaults(**{arg:val})

    args = p.parse_args()

    sheet_obj = Sheets()

    # Setting up timestamped directory for log, model, and other object storage
    try:
        args.job_num = os.environ['JOB_ID']
        log.info(f"Job ID = {args.job_num}")
    except:
        log.info("Job information not found")



    if args.timestamp is None:
        args.timestamp = time.strftime("run_%Y_%m_%d_%H_%M_%S") + f"_{args.language}_{args.job_num}_{socket.gethostname()}"

    else:
        args.keep_training=True
        log.info("Using timestamp: {0}".format(args.timestamp))
    if not args.keep_training:
        sheet_obj.add_run(vars(args))
    args.directory = os.path.join(args.root, args.timestamp)
    os.makedirs(args.directory, exist_ok=True)

    log.handlers.clear()
    formatter = logging.Formatter('%(message)s')
    fh = logging.FileHandler(os.path.join(args.directory, "log.txt"))
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    log.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    log.setLevel(logging.INFO)
    log.addHandler(ch)
    log.info(p.format_values())


    torch_utils.set_seed(cuda=args.use_cuda)
    p.write_config_file(args, [os.path.join(args.directory, 'config.ini')])

    if args.parameter_tune:
        args = parameter_tune(args)

    try:
        for i in range (torch.cuda.device_count()):
            torch_utils.gpu(torch.zeros((1)), gpu=args.use_cuda).to('cuda:{0}'.format(i))

        if torch.cuda.is_available():
            log.info("Using CUDA device:{0}".format(torch.cuda.current_device()))
        else:
            log.info("Not using CUDA :(")
        save_code(args)
        model(args)
        sheet_obj.update_run(vars(args), end=True)

    except KeyboardInterrupt:
        sheet_obj.error_run(vars(args), "KeyboardInterrupt", end_type="Aborted")
    except InterruptedError:
        sheet_obj.error_run(vars(args), "OS Interrupt", end_type="Aborted")
    except Exception as e:
        sheet_obj.error_run(vars(args), e)
        log.error(str(e))
        log.error(traceback.format_exc())
        raise e
