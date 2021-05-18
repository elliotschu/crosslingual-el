"""
Elliot Schumacher, Johns Hopkins University
Created 5/26/20
"""
from codebase import el_scorer
import os
import logging
import pandas as pd
def main():
    log = logging.getLogger()
    log.handlers.clear()
    formatter = logging.Formatter('%(message)s')

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    log.setLevel(logging.INFO)
    log.addHandler(ch)

    timestamp = "run_2020_06_24_10_37_42_ALL_8566794_r3n07"
    pred_fn = "t_-1.00.pred.tac"
    gold_file = f"/Users/elliotschumacher/Dropbox/git/clel/results/{timestamp}/eval.gold.tac"
    pred_file = f"/Users/elliotschumacher/Dropbox/git/clel/results/{timestamp}/eval_99/{pred_fn}"

    gold_nil_file = f"/Users/elliotschumacher/Dropbox/git/clel/results/{timestamp}/eval_nil.gold.tac"
    pred_nil_file = f"/Users/elliotschumacher/Dropbox/git/clel/results/{timestamp}/nil/{pred_fn}"
    os.makedirs(os.path.dirname(pred_nil_file), exist_ok=True)

    gold_non_file = f"/Users/elliotschumacher/Dropbox/git/clel/results/{timestamp}/eval_nonnil.gold.tac"
    pred_non_file = f"/Users/elliotschumacher/Dropbox/git/clel/results/{timestamp}/nonnil/{pred_fn}"
    os.makedirs(os.path.dirname(pred_non_file), exist_ok=True)


    header_list = ["system_id", "query_id", "mention_string", "doc_id_offsets", "link_id", "entity_type",
                   "mention_type",
                   "confidence", "web_search", "wiki_text", "unknown"]
    data_file = "/Users/elliotschumacher/Dropbox/git/clel/datasets/LDC2019T02/tac_kbp_2015_tedl_training_gold_standard_entity_mentions.tab"

    data = pd.read_csv(data_file, sep="\t", names=header_list, header=None, )


    with open (gold_file) as gold, open(pred_file) as pred, open (gold_nil_file, 'w') as gold_nil, \
        open(pred_nil_file, 'w') as pred_nil,  open(gold_non_file, 'w') as gold_non, open(pred_non_file, 'w') as pred_non:

        for gold_l , pred_l in zip(gold, pred):
            id, gold_kb = gold_l.strip().split(" ")
            row = data[data['query_id'] == id]
            if gold_kb == "NIL":
                gold_nil.write(gold_l)
                pred_nil.write(pred_l)


            else:
                gold_non.write(gold_l)
                pred_non.write(pred_l)



    nil = el_scorer.systemsRankingScript(goldStdFile= gold_nil_file,
                                   systemsDir= os.path.dirname(pred_nil_file), focusElFile=gold_nil_file)
    nonnil = el_scorer.systemsRankingScript(goldStdFile= gold_non_file,
                                   systemsDir= os.path.dirname(pred_non_file), focusElFile=gold_non_file)

    for lang in ["ENG", "SPA", "CMN"]:
        gold_nil_file = f"/Users/elliotschumacher/Dropbox/git/clel/results/{timestamp}/eval_nil_{lang}.gold.tac"
        pred_nil_file = f"/Users/elliotschumacher/Dropbox/git/clel/results/{timestamp}/nil_{lang}/{pred_fn}"
        os.makedirs(os.path.dirname(pred_nil_file), exist_ok=True)

        gold_non_file = f"/Users/elliotschumacher/Dropbox/git/clel/results/{timestamp}/eval_nonnil_{lang}.gold.tac"
        pred_non_file = f"/Users/elliotschumacher/Dropbox/git/clel/results/{timestamp}/nonnil_{lang}/{pred_fn}"
        os.makedirs(os.path.dirname(pred_non_file), exist_ok=True)

        with open(gold_file) as gold, open(pred_file) as pred, \
                open(gold_nil_file, 'w') as gold_nil, \
                open(pred_nil_file, 'w') as pred_nil, \
                open(gold_non_file, 'w') as gold_non, \
                open(pred_non_file, 'w') as pred_non:

            for gold_l, pred_l in zip(gold, pred):
                id, gold_kb = gold_l.strip().split(" ")
                row = data[data['query_id'] == id]
                this_lang = row['doc_id_offsets'].iloc[0][:3]
                if this_lang == lang:
                    if gold_kb == "NIL":
                        gold_nil.write(gold_l)
                        pred_nil.write(pred_l)

                    else:
                        gold_non.write(gold_l)
                        pred_non.write(pred_l)
        print(lang)
        nil = el_scorer.systemsRankingScript(goldStdFile=gold_nil_file,
                                             systemsDir=os.path.dirname(pred_nil_file), focusElFile=gold_nil_file)
        nonnil = el_scorer.systemsRankingScript(goldStdFile=gold_non_file,
                                                systemsDir=os.path.dirname(pred_non_file), focusElFile=gold_non_file)
    for lang in ["ENG", "SPA", "CMN"]:
        for type in ["DF", "NW"]:
            gold_nil_file = f"/Users/elliotschumacher/Dropbox/git/clel/results/{timestamp}/eval_nil_{lang}_{type}.gold.tac"
            pred_nil_file = f"/Users/elliotschumacher/Dropbox/git/clel/results/{timestamp}/nil_{lang}_{type}/{pred_fn}"
            os.makedirs(os.path.dirname(pred_nil_file), exist_ok=True)

            gold_non_file = f"/Users/elliotschumacher/Dropbox/git/clel/results/{timestamp}/eval_nonnil_{lang}_{type}.gold.tac"
            pred_non_file = f"/Users/elliotschumacher/Dropbox/git/clel/results/{timestamp}/nonnil_{lang}_{type}/{pred_fn}"
            os.makedirs(os.path.dirname(pred_non_file), exist_ok=True)

            with open(gold_file) as gold, open(pred_file) as pred, \
                    open(gold_nil_file, 'w') as gold_nil, \
                    open(pred_nil_file, 'w') as pred_nil, \
                    open(gold_non_file, 'w') as gold_non, \
                    open(pred_non_file, 'w') as pred_non:

                for gold_l, pred_l in zip(gold, pred):
                    id, gold_kb = gold_l.strip().split(" ")
                    row = data[data['query_id'] == id]
                    this_lang = row['doc_id_offsets'].iloc[0][:3]
                    this_type = row['doc_id_offsets'].iloc[0][4:6]
                    if this_lang == lang and this_type == type:
                        if gold_kb == "NIL":
                            gold_nil.write(gold_l)
                            pred_nil.write(pred_l)

                        else:
                            gold_non.write(gold_l)
                            pred_non.write(pred_l)
            print(lang)
            print(type)
            nil = el_scorer.systemsRankingScript(goldStdFile=gold_nil_file,
                                                 systemsDir=os.path.dirname(pred_nil_file), focusElFile=gold_nil_file)
            nonnil = el_scorer.systemsRankingScript(goldStdFile=gold_non_file,
                                                    systemsDir=os.path.dirname(pred_non_file), focusElFile=gold_non_file)


if __name__ == "__main__":
    main()