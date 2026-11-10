import argparse
import sys
from src import ImageDownloader

def main(arguments):
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", action="store_true")
    args = parser.parse_args(arguments[1:])
    if args.d:
        ImageDownloader.download_raise_nef("./RAISE_258.csv","./images", 100)
        return



if __name__ == "__main__":
    main(sys.argv)