import matplotlib.pyplot as plt
import numpy as np

def solve(filename):
    f = open('res/' + filename + '.txt')
    sizes = []
    times = []
    title = filename 
    while True:
        line = f.readline()
        if line:
            slices = line.split(" ")
            if len(slices) <= 2:
                break
            size = int(slices[0])
            time = float(slices[1])
            sizes.append(size)
            times.append(time)
    return title, sizes, times

if __name__ == '__main__':
    my_res = ['mmult_1','mmult_2','mmult_3','mmult_4','mmult_5','mmult_6', 'mmult_7', 'mmult_8']
    cmp_res = ['MMult_4x4_13','MMult_4x4_14']
    plt.xlabel('size')
    plt.ylabel('gflops')
    plt.subplots_adjust(right=0.7)
    for res in my_res: 
        t, x, y = solve(res)
        plt.plot(x, y, label=t)
    for res in cmp_res: 
        t, x, y = solve(res)
        plt.plot(x, y, label=t, linestyle='dashed')
    plt.legend(loc=3, bbox_to_anchor=(1.05, 0))
    plt.savefig('MY_MMult_res.png', dpi=400)
    plt.show()
