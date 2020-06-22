#!/usr/bin/env python3

import sys
import pickle

def main():
    file = open(sys.argv[1], "rb")
    data = pickle.load(file)

    print(data[0]["data"]["integration"])

if __name__ == "__main__":
    main()
