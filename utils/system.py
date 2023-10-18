import sys
import os
import re
import argparse

def parse_params():
    '''
    这个函数使用argparse库来解析命令行参数。它设置了一个参数-c或--clean-cache，
    如果用户在命令行中指定了这个参数，将会清理缓存文件。
    具体来说，它遍历一个名为"features"的文件夹，删除所有文件名以".npy"结尾的文件。
    然后，它检查"splits"文件夹中
    是否存在名为"hold_out_ids.txt"和"training_ids.txt"的文件，
    如果存在，也会将它们删除。
    最后，它打印 "All clear" 表示清理操作已完成。
    '''
    parser = argparse.ArgumentParser(description='FakeNewsChallenge fnc-1-baseline')
    parser.add_argument('-c', '--clean-cache', action='store_true', default=False, help="clean cache files")
    params = parser.parse_args()

    if not params.clean_cache:
        return

    dr = "features"
    for f in os.listdir(dr):
        if re.search('\.npy$', f):
            fname = os.path.join(dr, f)
            os.remove(fname)
    for f in ['hold_out_ids.txt', 'training_ids.txt']:
        fname = os.path.join('splits', f)
        if os.path.isfile(fname):
            os.remove(fname)
    print("All clear")


#检查Python版本
def check_version(): 
    if sys.version_info.major < 3:
        sys.stderr.write('Please use Python version 3 and above\n')
        sys.exit(1)
