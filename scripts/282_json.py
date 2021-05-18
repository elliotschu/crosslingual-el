"""
Elliot Schumacher, Johns Hopkins University
Created 3/6/20
"""
import json

def main():
    fn = "/Users/elliotschumacher/Downloads/ru-en.json"
    with open(fn) as f:
        for line in f:
            jobj = json.loads(line)
            print(jobj)


if __name__ == "__main__":
    main()