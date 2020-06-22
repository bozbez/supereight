#!/usr/bin/env python3

import pickle
import sys

def main():
    file = open(sys.argv[1], "rb")
    data = pickle.load(file)

    print("sequence,tracking,integration,raycasting,other")

    for run in data:
        d = run["data"]

        sequence = run["sequence"]

        tracking = float(d["tracking"]["mean"])
        integration = float(d["integration"]["mean"])
        raycasting = float(d["raycasting"]["mean"])

        computation = float(d["computation"]["mean"])

        other = computation - tracking - integration - raycasting

        print(f"{sequence},{tracking},{integration},{raycasting},{other}")

if __name__ == "__main__":
    main()
