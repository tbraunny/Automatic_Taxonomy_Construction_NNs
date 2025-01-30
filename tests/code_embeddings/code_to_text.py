import os 
import glob

def convert(path):
    with open(path , "r") as f:
        code = f.read()

    #with open()

def main():
    ann_name = "alexnet"
    path = glob.glob(f"/home/richw/tom/ATCNN/data/{ann_name}/*.py")
    convert(path)

if __name__ == '__main__':
    main()