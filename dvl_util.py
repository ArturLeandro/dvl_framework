import copy
import matplotlib.pyplot as plt
import pandas as pd

def generate_reference_points(num_objs, num_divisions_per_obj=4):
    '''Generates reference points for NSGA-III selection. This code is based on
    `jMetal NSGA-III implementation <https://github.com/jMetal/jMetal>`_.
    '''
    def gen_refs_recursive(work_point, num_objs, left, total, depth):
        if depth == num_objs - 1:
            work_point[depth] = left/total
            ref = copy.deepcopy(work_point)
            return [ref]
        else:
            res = []
            for i in range(left+1):
                work_point[depth] = i/total
                res = res + gen_refs_recursive(work_point, num_objs, left-i, total, depth+1)
            return res
    return gen_refs_recursive([0]*num_objs, num_objs, num_divisions_per_obj, num_divisions_per_obj, 0)

def truncate(s, lower, upper):
    for i in range(len(s)):
        if s[i] < lower[i]:
            s[i] = lower[i]
        elif s[i] > upper[i]:
            s[i] = upper[i]
    return s

def plotObjSpace(objectives, reference_points):
    xs = objectives[:,0]
    ys = objectives[:,1]
    zs = objectives[:,2]

    rxs = reference_points[:,0]
    rys = reference_points[:,1]
    rzs = reference_points[:,2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(xs, ys, zs, color='blue')
    ax.scatter3D(rxs, rys, rzs, color='orange')

    ax.set_xlabel('Objective X')
    ax.set_ylabel('Objective Y')
    ax.set_zlabel('Objective Z')

    plt.show()


def solNSGA(n):
    pd_solutions = pd.read_csv('teste.csv', header=None, sep=',') 
    solutions = pd_solutions.to_numpy()
    solutions = solutions.copy(order='C')
    return solutions[:n,:]