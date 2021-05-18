"""
Elliot Schumacher, Johns Hopkins University
Created 5/5/20
"""
import configargparse
import shutil
import os
import pandas as pd
def main():

    p = configargparse.ArgParser()
    p.add('--input', help='root directory for storing results',default='/Users/elliotschumacher/Dropbox/git/clel/datasets/LDC2005T34/text')
    p.add('--output', help='root directory for storing results',default='/Users/elliotschumacher/Dropbox/git/clel/datasets/LDC2005T34/name_pairs.csv')

    args, unknown = p.parse_known_args()
    output_list = []
    for filename in os.listdir(args.input):
        if "_ce_" in filename:
            with open(os.path.join(args.input, filename), 'rb') as f_gb:
                for i, line in enumerate(f_gb):
                    line_split = line.split(b"\t")
                    try:
                        chinese = line_split[0].decode("gb2312")
                        chinese_conversion = chinese.encode('utf8').decode('utf8')
                        eng = line_split[1].decode('ascii')
                        eng = eng.replace("/", " ")
                        output_list.append({
                            "ENG" : eng,
                            "L2" : chinese_conversion
                        })
                        output_str = f"{chinese_conversion}\t{eng}\n"
                    except Exception as e:
                        print(f"{filename}\t{i}\t{line_split}")
    name_df = pd.DataFrame().from_dict(output_list)
    name_df.to_csv(args.output, index=False)

if __name__ == "__main__":
    main()