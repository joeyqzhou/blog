## 在pytorch中load超大训练数据



> by joeyqzhou

### 最简单方式:

1 单线程获取数据到内存中

2  train的过程

```python
   for epoch in range(num_epochs):
        for i in range(inst_size): 
        		#截取 batch_x, batch_y
          	#batch_x, batch_y 转换为tensor
            #model.forward()
            #loss.backward()
            #optimizer.step()
                
```



这种方式代码简单。缺点load数据过慢，数据全部存储在内存当中。

当训练数据过大的时候load很慢，内存会溢出



###  多进程load数据



如下是一个多进程load数据的例子

```python
from multiprocessing import Pool

def process_line(line):
    return "FOO: %s" % line

if __name__ == "__main__":
    pool = Pool(4)
    file = "train.txt" #你的输入数据
    ret = []
    with open(file) as source_file:
        # chunk the work into batches of 4 lines at a time
        results = pool.map(process_line, source_file, 4)

    print(results)
```



当并发处理函数，是在另外的函数内，需要输入参数时，可以参考如下实现

```python
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
    lines = ["aa", "bb", "cc"]
    print(multi_process_line(lines, "FOO"))
```



拉取数据可以利用pytorch自带的工具 torch.utils.data.Dataset， 这样可以实现：多进程/并行拉取（一边训练一边拉取),   init函数进行初始化，读取数据。

getitem是返回按index拉取的函数，len表示这个iterator有多长，什么时候结束

```python
#Dataset.py
import torch
from multi_process_argument import multi_process_line

class Dataset(torch.utils.data.Dataset):
    def __init__(self, file, prefix):
        f = open(file)
        self.data = multi_process_line(f, prefix)
        self.len = len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len
```



调用举例. 注意为了提升性能，如果内存足够，建议pin_memory=True, num_workers是进程数量，如果batch_size不足时会报错最后一轮的数据就不要了drop_last = true

```python
import torch
from Dataset import Dataset

arg = "HOO"
train_set = Dataset("train.txt", arg)
params = {'batch_size': 2,
              'shuffle': True,
              'pin_memory': True,
              'num_workers': 8,
              'drop_last' : True}

training_generator = torch.utils.data.DataLoader(train_set, **params)

for x in training_generator:
    print(x)
    print("-----")
```





### 解决load超大训练数据问题

如果训练数据过大，就不能把数据全部存储在内存中，否则会内存溢出

如何解决？

思路一，把数据切分为多个小文件， load一个, train一下 , 这样理论上就不会占用内存了.

但是实际我跑出来内存还是随着训练会不断提升，这个原因还没有找到。已经尝试了del相关变量等方法还是不行。



```python
files = glob.glob(os.path.join(train_file_dir))
for epoch in range(num_epochs):
  for i, file in enumerate(files):
    train_set = Dataset(file,  arg)
    for x,y in train_set:
      ....
```



思路二：

使用 torch.utils.data.IterableDataset, 类似如下。Dataloader和上面的类似。有一个区别num_workers要设置为0， 否则有重复数据，设置为N, 那数据就重复N次。

```python
class IterDataset(torch.utils.data.IterableDataset):
    def __init__(self, arg):
        self.data_iter = 获取一个iterator

    def __iter__(self):
        return self.data_iter
```



所以例子如下，注意每一次epoch需要重新初始化Dataset, 不然它就是结束的状态。

```python
 params = {'batch_size': batch_size,
              'num_workers': 0,
              'pin_memory': True,
              'drop_last' : True}
  
  for epoch in range(num_epochs):
        train_set = IterDataset(train_file_dir,  arg)
        training_generator = torch.utils.data.DataLoader(train_set, **params)
        for batch_x, batch_y in training_generator:
          		...
```

