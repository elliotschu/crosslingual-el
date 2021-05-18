"""
Elliot Schumacher, Johns Hopkins University
Created 4/17/20
"""
from pymongo import MongoClient
from pprint import pprint
import configargparse
from collections import defaultdict
import random
import pickle
import traceback
import pandas
import re
def add(sent, mentions, lang):
    for link in sent.get("links"):
        if "title_ll" in link:
            mentions = add_mention(sent, mentions, link, lang)
    return mentions

def add_mention(sent, mentions, link, lang, title=None):
    link["sent"] = sent
    if title is None:
        mentions[lang][sent.get("title")][(sent.get("_id"), link["id"])] = link
    else:
        mentions[lang][title][(sent.get("_id"), link["id"])] = link

    return mentions

def main():
    # connect to MongoDB, change the << MONGODB URL >> to reflect your own connection string
    p = configargparse.ArgParser()
    p.add('--mongo_url')
    p.add('--output', default="wiki_candgen")
    p.add('--lang', default="ar")
    p.add("--cand_file", default="wiki.pkl.ar.csv.out")


    args, unknown = p.parse_known_args()
    client = MongoClient(args.mongo_url)
    db = client.uiuc_db

    en_pages = {}
    candidates = pandas.read_csv(args.cand_file)
    for i, row in candidates.iterrows():
        try:

            page_title = row['cand_title']\
                .replace("_", " ")
            if page_title != "NULLTITLE":
                page = db['uiuc_en_pages'].find_one(
                    {"title": page_title},
                )
                if page is not None:
                    en_pages[row['cand_title']] = page
                else:
                    print(f"Cannot find {page_title}")

        except Exception as e:
            print(e)
            print("Error for {0}".format(row))
            print(traceback.format_exc())

    with open(f"{args.output}_{args.lang}.pkl", 'wb') as out_f:
        pickle.dump(en_pages, out_f)

if __name__ == "__main__":
    main()
