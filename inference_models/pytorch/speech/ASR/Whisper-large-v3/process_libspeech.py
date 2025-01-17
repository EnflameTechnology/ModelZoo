import os
import os.path as osp
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='', help='dataset path.')
    parser.add_argument('--save_path', default='', help='save path.')
    return parser.parse_args()

def main():
    args = parse_arguments()
    save_txt = open(osp.join(args.save_path,'dev_clean_test.txt'),'w',encoding = "utf-8")
    for root, dirs, files in os.walk(args.data_path):
        for file in files:
            file_path = os.path.join(root, file)
            if file_path.find('.txt')!=-1:
                ori_txt = open(file_path,'r',encoding = "utf-8")
                for line in ori_txt.readlines():
                    str_ = line.split(' ')[0]
                    new_line=line.replace(str_,str_+'.flac,')
                    full_line = root+'/'+ new_line
                    save_txt.write(full_line)

if __name__ == '__main__':
    main()