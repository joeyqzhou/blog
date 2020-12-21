from multiprocessing import Pool
from functools import partial

def process_line(prefix, line):
    return prefix + ": %s" % line 

def multi_process_line(lines, prefix): 
    partial_process_line = partial(process_line, prefix) 
    threads = 4 
    pool = Pool(threads)

    results = pool.map(partial_process_line, lines)
    return results

if __name__ == "__main__":
    lines = open("train.txt") 
    print(multi_process_line(lines, "FOO")) #这里的lines是一个itertable daata就可以
