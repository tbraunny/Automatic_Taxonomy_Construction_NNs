import os 
import glob

"""
DEPRECATED
Convert python code to text for passage to RAG as document objects
"""

class CodeProcessor():
    def convert(path , count):
        with open(path , "r") as f:
            code = f.read()

        txt_path = path.replace(".py" , f"_txt_{count}.txt")
        with open(txt_path , "w") as f:
            f.write(code)

        print(f"Converted {os.path.basename(path)} to {os.path.basename(txt_path)} in location: {txt_path}")

def main():
    ann_name = "alexnet"
    path = glob.glob(f"/home/richw/tom/ATCNN/data/{ann_name}/*.py")

    count = 0
    for file in path:
        CodeProcessor.convert(file , count)
        count += 1

if __name__ == '__main__':
    main()