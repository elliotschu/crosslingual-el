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
import os
import re

def add_mention(sent, mentions, link, lang, title):
    this_ment = {"sent" : sent, "link" : link}
    mentions[lang][title][(sent.get("_id"), link["id"])] = this_ment


    return mentions

def process_sentence(sent, mention_counts, out_mentions_by_count, in_mentions, out_mentions, lang, page_title):
    for link in sent.get("links"):
        if "title_ll" in link:
            if "title_ll" in link and link["title_ll"] == page_title:
                in_mentions = add_mention(sent, in_mentions, link, lang, title=sent.get("title"))
                mention_counts += 1
            elif "title_ll" in link:
                out_mentions = add_mention(sent, out_mentions, link, lang, title=link.get("title_ll"))

                out_mentions_by_count[link.get("title_ll")] += 1
    return in_mentions, out_mentions, out_mentions_by_count

def main():
    # connect to MongoDB, change the << MONGODB URL >> to reflect your own connection string
    p = configargparse.ArgParser()
    p.add('--mongo_url')
    p.add('--output_dir', default="/Users/elliotschumacher/Dropbox/git/clel/data")
    p.add('--n', type=int, default=10000)
    p.add('--min_mentions', type=int, default=5)
    p.add('--max_mentions', type=int, default=50)
    p.add('--max_pages', type=int, default=20)
    p.add('--languages', default='ko')

    p.add('--perc_nil', type=float, default=0.2)
    p.add("--pages", default="pages.txt")

    args, unknown = p.parse_known_args()
    client = MongoClient(args.mongo_url)
    db = client.uiuc_db

    languages = args.languages.split(",")

    en_pages = {}
    in_mentions = defaultdict(lambda: defaultdict(dict))
    out_mentions = defaultdict(lambda: defaultdict(dict))
    out_mentions_by_count = defaultdict(lambda : 0)
    l2_pages = defaultdict(lambda: defaultdict())
    page_file = open(args.pages)
    mention_counts = 0
    total_mention_counts = 0
    max_frequency = 0
    lang_counts = defaultdict(lambda : 0)
    for l in page_file:
        try:
            l_split = l.split("\t")
            page_title = l_split[0].strip()[1:-1]\
                .replace("_", " ")\
                .replace("$002E", ".")\
                .replace("$0027", "'")\
                .replace("$0028", "(")\
                .replace("$0029", ")")\
                .replace("$002C", ",")
            page_freq = int(l_split[-1])
            max_frequency = max(page_freq, max_frequency)
            # step 0 - get english page
            page = db['uiuc_en_pages'].find_one(
                {"title": page_title},
            )
            if page is not None:
                en_pages[page_title] = page
            else:
                for char in re.finditer(r"\$([0-9A-F][0-9A-F][0-9A-F][0-9A-F])", page_title):
                    char_int = int(char.group()[1:], 16)
                    # print(char_int)
                    # print(chr(char_int))
                    page_title = page_title.replace(char.group(), chr(char_int))
                page = db['uiuc_en_pages'].find_one(
                    {"title": page_title},
                )
                en_pages[page_title] = page

            # step 1 - get frequencies for sampling across languages
            lang_in_freqs = {lang: random.randrange(0, page_freq) for lang in languages}

            for lang in languages:
                # sample for mention according to distribution
                sentence_cursor_nonnil = db['uiuc_{0}'.format(lang)].aggregate([
                     {'$match': {"links.title_ll": page_title}},
                     {'$sample': {'size': lang_in_freqs[lang]}}
                ])

                for sent in sentence_cursor_nonnil:

                    in_mentions, out_mentions, out_mentions_by_count = process_sentence(sent,
                                                                                        mention_counts,
                                                                                        out_mentions_by_count,
                                                                                        in_mentions,
                                                                                        out_mentions,
                                                                                        lang,
                                                                                        page_title)
                    inpage_mention_cursor_nonnil = db['uiuc_{0}'.format(lang)].aggregate([
                        {'$match': {"title": sent.get("title"), "len_links_ll": {"$gt": 0}}},])

                    for sent2 in inpage_mention_cursor_nonnil:
                        in_mentions, out_mentions, out_mentions_by_count = process_sentence(sent2,
                                                                                            mention_counts,
                                                                                            out_mentions_by_count,
                                                                                            in_mentions,
                                                                                            out_mentions,
                                                                                            lang,
                                                                                            page_title)

                total_mention_counts += mention_counts
                lang_counts[lang] += mention_counts
                mention_counts = 0

        except Exception as e:
            print(e)
            print("Error for {0}".format(l))
            print(traceback.format_exc())
    for l in languages:
        current = sum(len(x.keys()) for x in in_mentions[l].values())
        print("Stage 0 - {0}:{1}".format(l, current))
    # get some out mentions
    frequency_list = [random.randrange(1, max_frequency) for _ in range(len(en_pages) * 2)]
    mention_counts = 0

    for freq in frequency_list:
        title = random.choice(list(out_mentions_by_count.keys()))
        del out_mentions_by_count[title]
        # get eng page
        page = db['uiuc_en_pages'].find_one(
            {"title": title},
        )
        en_pages[title] = page

        for lang in languages:
            # move out mentions to in
            for i, ((s_id, m_id), m) in enumerate(out_mentions[lang][title].items()):
                in_mentions = add_mention(m["sent"], in_mentions, m["link"], lang, title=m["sent"].get("title"))
                mention_counts += 1
            total_mention_counts += mention_counts
            lang_counts[lang] += mention_counts
            mention_counts = 0

    for l in languages:
        current = sum(len(x.keys()) for x in in_mentions[l].values())
        print("Stage 1 - {0}:{1}".format(l, current))

    for l in languages:
        for l2_title in in_mentions[l]:
            if not l2_title in l2_pages[l]:
                l2_pages[l][l2_title] = db['uiuc_{0}_pages'.format(l)].find_one(
                    {"title": l2_title}, )

    nil_mentions = defaultdict(lambda: defaultdict(dict))
    for lang in languages:
        num_nil = int(lang_counts[lang] * args.perc_nil)
        print("Adding {0} NIL".format(num_nil))
        num_added = 0
        for j in range(num_nil):
            title = random.choice(list(out_mentions[lang].keys()))
            q = -1
            for q, (m_id, m) in enumerate(out_mentions[lang][title].items()):
                nil_mentions[lang][m['sent'].get("title")][(m["link"]["id"], m["sent"].get("_id"))] = m
                num_added += 1
            if num_added > num_nil:
                break
    for l in languages:
        current = sum(len(x.keys()) for x in nil_mentions[l].values())
        print("NIL - {0}:{1}".format(l, current))



    l2_pages = {k: dict(v) for k,v in l2_pages.items()}
    mentions = {k: dict(v) for k,v in in_mentions.items()}
    nil_mentions = {k: dict(v) for k,v in nil_mentions.items()}

    mention_list = []
    nil_mention_list = []
    for lang in languages:
        this_lang = []
        for title in mentions[lang]:
            for m in mentions[lang][title].values():
                #pprint(m)
                m_dict = {
                    "mention": m["link"]["text"],
                    "lang" : lang,
                    "page_title": title,
                    "link_title": m["link"]["title"],
                    "m_id" : m["link"]["id"],
                    "sent_id" : m["sent"].get("_id"),
                    "title_ll_link" : m["link"]["title_ll"],

                }
                for subfield in m:
                    if type(m[subfield]) is str:
                        m_dict[subfield] = m[subfield]
                mention_list.append(m_dict)
                this_lang.append(m_dict)
        for title in nil_mentions[lang]:
            for m in nil_mentions[lang][title].values():
                #pprint(m)
                m_dict = {
                    "mention" : m["link"]["text"],
                    "lang": lang,
                    "page_title": title,
                    "link_title": m["link"]["title"],
                    "m_id": m["link"]["id"],
                    "sent_id": m["sent"].get("_id"),
                    "title_ll_link": m["link"]["title_ll"],

                }
                for subfield in m["link"]:
                    if type(m["link"][subfield]) is str:
                        m_dict[subfield] = m["link"][subfield]
                nil_mention_list.append(m_dict)
                this_lang.append(m_dict)
        lang_df = pandas.DataFrame().from_dict(this_lang)
        os.makedirs(os.path.join(args.output_dir, lang), exist_ok=True)

        lang_df.to_csv(os.path.join(args.output_dir, lang, f'mention.{lang}.csv'), index=False)

        # mention_df = pandas.DataFrame().from_dict(mention_list)
        # mention_df.to_excel("{0}.xlsx".format(args.output))
        # mention_df.to_csv("{0}.csv".format(args.output))
        #
        # mention_df = pandas.DataFrame().from_dict(nil_mention_list)
        # mention_df.to_excel("{0}.nil.xlsx".format(args.output))
        # mention_df.to_csv("{0}.nil.csv".format(args.output))

        with open(os.path.join(args.output_dir, lang, f'wiki_{lang}.pkl'), 'wb') as out_f:
            pickle.dump({
                "en_pages" : en_pages,
                "mentions" : mentions,
                "nil_mentions": nil_mentions,
                "l2_pages" : l2_pages,
            }, out_f)

if __name__ == "__main__":
    main()
