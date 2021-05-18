"""
Elliot Schumacher, Johns Hopkins University
Created 2/18/20
"""
from codebase.sheets import Sheets
import configargparse
import shutil
import os
def main():
    sheet_obj = Sheets()
    sheet = sheet_obj.get_sheet()
    directories = sheet.col_values(1)[1:]

    print(directories)
    p = configargparse.ArgParser()
    p.add('-c', '--my-config', is_config_file=True, help='config file path')
    p.add('--root', help='root directory for storing results',default='./results')
    p.add('--file_size', help='root directory for storing results',default=500)

    args, unknown = p.parse_known_args()

    to_be_removed_dirs = []
    print("checking :{0}".format(args.root))
    paperlist = set(
        [item for sublist in sheet_obj.get_sheet("paper_2021").get_all_values() for item in sublist]
    )
    print([x for x in paperlist if "_" in x])
    for subdir in os.listdir(args.root):
        if subdir not in directories:
            is_in_paper_list = subdir in paperlist
            to_be_removed_dirs.append(subdir)
            resp = input(f"Delete {subdir}, in paper {is_in_paper_list}?")
            if resp.upper() == "Y":
                shutil.rmtree(os.path.join(args.root, subdir))
                print("Deleted {0}".format(subdir))
    print("Processing large files!!!!!!!!")
    save_chkpts = sheet_obj.get_sheet("save_chkpt")
    save_dirs = [f.replace("/", "") for f in save_chkpts.col_values(1)]
    print(save_dirs)
    files_by_time = sorted(
        [os.path.join(args.root, f) for f in os.listdir(args.root) if f not in save_dirs],
        key=os.path.getctime,
    )

    files_excluded = [ f for f in os.listdir(args.root) if f in save_dirs]



    print(f"Files to be excluded : {len(save_dirs)}")

    print(f"Files excluded : {len(files_excluded)}")

    diff = set(files_excluded).symmetric_difference(set(save_dirs))

    print("diff")
    print(diff)



    for subdir in files_by_time:
        if os.path.isdir(subdir):
            for file in os.listdir(subdir):
                size = os.path.getsize(os.path.join(subdir, file)) / 1024.0 / 512.0
                if size >= args.file_size:
                    basename = os.path.split(subdir)[-1]
                    is_in_paper_list = basename in paperlist
                    if is_in_paper_list and file.endswith(".tar.gz"):
                        print(f"Skipping {os.path.join(subdir, file)}, in paper")
                    else:
                        resp = input(
                            f"Delete {os.path.join(subdir, file)}, paper = {is_in_paper_list}?"
                        )
                        if resp.upper() == "Y":
                            os.remove(os.path.join(subdir, file))
                            print("Deleted {0}".format(os.path.join(subdir, file)))


if __name__ == "__main__":
    main()