#!/usr/bin/python3

import pickle
import sys

def main():
    file = open(sys.argv[1], "rb")
    data = pickle.load(file)

    print(data[0]["data"]["integration"])

if __name__ == "__main__":
    main()
