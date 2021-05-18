"""
Elliot Schumacher, Johns Hopkins University
Created 5/26/20
"""

import pandas as pd
import os
def main():
    timestamp = "run_2020_06_03_10_01_31_ALL_8514392_r3n07"
    pred_fn = "t_-1.00.pred.tac"
    gold_file = f"/Users/elliotschumacher/Dropbox/git/clel/results/{timestamp}/eval.gold.tac"
    pred_file = f"/Users/elliotschumacher/Dropbox/git/clel/results/{timestamp}/eval_249/{pred_fn}"

    lang = "SPA"

    gold_lang_file = f"/Users/elliotschumacher/Dropbox/git/clel/results/{timestamp}/eval_{lang}.gold.tac"
    pred_lang_file = f"/Users/elliotschumacher/Dropbox/git/clel/results/{timestamp}/{lang}/{pred_fn}"
    os.makedirs(f"/Users/elliotschumacher/Dropbox/git/clel/results/{timestamp}/{lang}", exist_ok=True)

    data_file = "/Users/elliotschumacher/Dropbox/git/clel/datasets/LDC2019T02/tac_kbp_2015_tedl_eval_gold_standard_entity_mentions.tab"
    header_list = ["system_id", "query_id", "mention_string", "doc_id_offsets", "link_id", "entity_type",
                   "mention_type",
                   "confidence", "web_search", "wiki_text", "unknown"]
    data = pd.read_csv(data_file, sep="\t", names=header_list, header=None, )

    with open (gold_file) as gold, open(pred_file) as pred, open (gold_lang_file, 'w') as gold_lang, \
        open(pred_lang_file, 'w') as pred_lang:

        for gold_l , pred_l in zip(gold, pred):
            id, gold_kb = gold_l.strip().split(" ")

            this_mention = data[data['query_id'] == id]

            if this_mention["doc_id_offsets"].iloc[0].startswith(lang):
                gold_lang.write(gold_l)
                pred_lang.write(pred_l)


if __name__ == "__main__":
    main()