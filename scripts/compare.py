"""
Elliot Schumacher, Johns Hopkins University
Created 4/9/20
"""
import pandas as pd
import os
def main():
    file1 = "/Users/elliotschumacher/Dropbox/git/clel/results/run_2020_04_08_22_29_11_ALL_r3n03/eval_199.xlsx"
    #file2 should be the ZS version
    file2 = "/Users/elliotschumacher/Dropbox/git/clel/results/run_2020_04_07_21_56_54_ALL_r7n01/eval_249.xlsx"
    data_file = "/Users/elliotschumacher/Dropbox/git/clel/datasets/LDC2019T02/ENG/tac_kbp_2015_tedl_training_gold_standard_entity_mentions.tab"
    header_list = ["system_id", "query_id", "mention_string", "doc_id_offsets", "link_id", "entity_type", "mention_type",
                   "confidence", "web_search", "wiki_text", "unknown"]
    df_a = pd.read_excel(file1)
    df_b = pd.read_excel(file2)
    data = pd.read_csv(data_file, sep="\t", names=header_list, header=None,)

    same = ["_text",
            "_sentence",
            "_gold_kbid",
            "_num_kb_cands",
            "~~mention_uuid",
            "~~comm",
            "_gold_name",
            ]
    new_columns = ["lang", "train_eng", "train_cmn", "train_spa"]
    column_renames_a = {}
    column_renames_b = {}
    new_columns.extend(same)

    for col in df_a.columns:
        if col not in same:
            new_columns.append(col + "_A")
            new_columns.append(col + "_B")
            column_renames_a[col] = col + "_A"
            column_renames_b[col] = col + "_B"

    df_a = df_a.rename(columns=column_renames_a)
    df_b = df_b.rename(columns=column_renames_b)


    df_both_correct = pd.DataFrame(columns=new_columns)
    df_a_correct = pd.DataFrame(columns=new_columns)
    df_b_correct = pd.DataFrame(columns=new_columns)
    df_none_correct = pd.DataFrame(columns=new_columns)
    eval_mentions = set()
    for i, row_a in df_a.iterrows():
        eval_mentions.add(row_a['~~mention_uuid'])

    data = data[~data.query_id.isin(eval_mentions)]
    data = data[~data['link_id'].str.startswith('NIL')]
    data['lang'] = data.apply(lambda row: row.doc_id_offsets[:3], axis = 1)

    link_lang_pivot = pd.pivot_table(data, index=['link_id', 'lang'], values='query_id', aggfunc=len).reset_index()
    link_lang_results = []
    for i, row_a in df_a.iterrows():
        if row_a['_gold_kbid'] != 'NIL':
            row_b = df_b[df_b['~~mention_uuid'] == row_a['~~mention_uuid']].iloc[0]

            for s_col in same:
                row_b = row_b.drop(s_col)
            new_row = pd.concat([row_a, row_b])
            #/exp/eschumacher/clel_data/LDC2019T02/SPA/training/SPA_DF_000406_20140930_F000000BA.df.xml
            new_row['lang'] = os.path.basename(new_row["~~comm"])[:3]

            a_correct = row_a['accuracy_A'] == 1
            b_correct = row_b['accuracy_B'] == 1

            if a_correct == 1 and b_correct:
                df_both_correct = df_both_correct.append(new_row)
            elif a_correct and not b_correct:
                df_a_correct = df_a_correct.append(new_row)
            elif not a_correct and b_correct:
                df_b_correct = df_b_correct.append(new_row)
            elif not a_correct and not b_correct:
                df_none_correct = df_none_correct.append(new_row)

            eng_num = link_lang_pivot[(link_lang_pivot['link_id'] == row_a['_gold_kbid']) & (link_lang_pivot['lang'] == 'ENG')]
            cmn_num = link_lang_pivot[(link_lang_pivot['link_id'] == row_a['_gold_kbid']) & (link_lang_pivot['lang'] == 'CMN')]
            spa_num = link_lang_pivot[(link_lang_pivot['link_id'] == row_a['_gold_kbid']) & (link_lang_pivot['lang'] == 'SPA')]

            if len(eng_num['query_id']) > 0:
                new_row["train_eng"] = eng_num['query_id'].iloc[0]
            else:
                new_row["train_eng"] = 0

            if len(cmn_num['query_id']) > 0:
                new_row["train_cmn"] = cmn_num['query_id'].iloc[0]
            else:
                new_row["train_cmn"] = 0

            if len(spa_num['query_id']) > 0:
                new_row["train_spa"] = spa_num['query_id'].iloc[0]
            else:
                new_row["train_spa"] = 0
            link_lang_row = {"a_correct": a_correct,
                             "b_correct": b_correct,
                             "lang": new_row['lang'],
                             "train_eng": new_row["train_eng"],
                             "train_spa": new_row["train_spa"],
                             "train_cmn": new_row["train_cmn"]}
            link_lang_results.append(link_lang_row)

    link_lang_results_df = pd.DataFrame.from_dict(link_lang_results)

    options = {}
    options['strings_to_formulas'] = False
    options['strings_to_urls'] = False

    with pd.ExcelWriter(os.path.join(os.path.dirname(file1), 'output.xlsx'), engine='xlsxwriter', options=options) as writer:
        df_both_correct.to_excel(writer, sheet_name='all_correct' ,index=False, freeze_panes=(1,0))

        df_a_correct.to_excel(writer, sheet_name='a_not_b' ,index=False, freeze_panes=(1,0))

        df_b_correct.to_excel(writer, sheet_name='b_not_a' ,index=False, freeze_panes=(1,0))

        df_none_correct.to_excel(writer, sheet_name='incorrect' ,index=False, freeze_panes=(1,0))



if __name__ == "__main__":
    main()