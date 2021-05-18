"""
Elliot Schumacher, Johns Hopkins University
Created 4/2/20
"""
import os
import pandas as pd
from collections import defaultdict
import pprint
def main():
    pass
    filename = "/Users/elliotschumacher/Dropbox/git/clel/results/run_2020_04_01_22_07_09_ALL_r2n06/eval_199.xlsx"
    pprinter = pprint.PrettyPrinter()

    all_lang = pd.read_excel(filename)
    kbid_across = defaultdict(lambda: set())
    kbid_counts = defaultdict(lambda: 0)
    kbid_accuracy = defaultdict(lambda: 0)
    kbid_subaccuracy = defaultdict(lambda: defaultdict(lambda: 0.))
    kbid_subcount = defaultdict(lambda: defaultdict(lambda: 0.))

    for i, mention_data in all_lang.iterrows():
        kbid = mention_data["_gold_kbid"]
        if kbid != 'NIL' and mention_data["no_cands"] == 0:
            filename = mention_data["~~comm"]
            lang, _ = os.path.basename(filename).split("_", 1)
            kbid_across[kbid].add(lang)
            kbid_counts[kbid] += 1
            kbid_accuracy[kbid] += mention_data["accuracy"]
            kbid_subcount[kbid][lang] += 1.
            kbid_subaccuracy[kbid][lang] += mention_data["accuracy"]

    lang_counts = defaultdict(lambda: 0)
    lang_total = defaultdict(lambda: 0)
    lang_accuracy = defaultdict(lambda: 0.)
    lang_subaccuracy = defaultdict(lambda: defaultdict(lambda: 0.))
    lang_subcount = defaultdict(lambda: defaultdict(lambda: 0.))

    for kbid, langs in kbid_across.items():
        lang_str = "_".join(sorted(langs))
        lang_counts[lang_str] += 1
        lang_total[lang_str] += kbid_counts[kbid]
        lang_accuracy[lang_str] += kbid_accuracy[kbid]
        for lang in kbid_subaccuracy[kbid]:
            lang_subaccuracy[lang_str][lang] += kbid_subaccuracy[kbid][lang]
            lang_subcount[lang_str][lang] += kbid_subcount[kbid][lang]
    column_dir = defaultdict(lambda: [])
    for lang in lang_counts:
        column_dir["lang"].append(lang)
        column_dir["unique"].append(lang_counts[lang])
        column_dir["total"].append(lang_total[lang])
        column_dir["correct"].append(lang_accuracy[lang])
        column_dir["accuracy"].append(lang_accuracy[lang] / lang_total[lang])
        for sl in ["ENG", "SPA", "CMN"]:
            column_dir["{0}_total".format(sl)].append(lang_subcount[lang][sl])
            column_dir["{0}_correct".format(sl)].append(lang_subaccuracy[lang][sl])
            if lang_subcount[lang][sl] > 0:
                column_dir["{0}_accuracy".format(sl)].append(lang_subaccuracy[lang][sl] / lang_subcount[lang][sl])
            else:
                column_dir["{0}_accuracy".format(sl)].append(0.0)
    results = pd.DataFrame.from_dict(column_dir)
    print(results.to_csv())
if __name__ == "__main__":
    main()