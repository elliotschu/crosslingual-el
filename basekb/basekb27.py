"""
Elliot Schumacher, Johns Hopkins University
Created 12/17/19
"""
import basekb.fbtools as fbt
import pickle

def main():
    fbt.configure(home='/Users/elliotschumacher/Dropbox/git/clel/tools/freebase-tools-1.2.0', config='config.dat')
    fbi = fbt.FreebaseIndex()
    fbi.describe()

if __name__ == "__main__":
    main()