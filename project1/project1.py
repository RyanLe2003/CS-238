import sys

import networkx as nx
import pandas as pd

from scipy.special import gammaln
from collections import defaultdict
from collections import Counter
import math

def write_gph(dag, idx2names, filename):
    with open(filename, 'w') as f:
        for edge in dag.edges():
            f.write("{}, {}\n".format(idx2names[edge[0]], idx2names[edge[1]]))

def read_csv(path):
    df = pd.read_csv(path)
    names = list(df.columns)
    for c in names:  # HERE
        df[c] = df[c].astype(int)
    max_val = {c: int(df[c].max()) for c in names}
    return df, names, max_val


# Local Bayesian score (Dirichlet-multinomial with αijk=1)
def local_bd_score(df, child, parents, r):
    ri = r[child]
    if not parents:
        counts = df[child].value_counts().reindex(range(1, ri+1), fill_value=0).values
        nij = counts.sum()
        return float(gammaln(ri) - gammaln(ri + nij) + (gammaln(1.0 + counts)).sum())
    total = 0.0
    gb = df.groupby(parents, observed=False, sort=False)
    for _, grp in gb:
        counts = grp[child].value_counts().reindex(range(1, ri+1), fill_value=0).values
        nij = counts.sum()
        total += float(gammaln(ri) - gammaln(ri + nij) + (gammaln(1.0 + counts)).sum())
    return total

def k2(df, names, r):
    name2idx = {nm: i for i, nm in enumerate(names)}
    dag = nx.DiGraph()
    dag.add_nodes_from(range(len(names)))
    parents = {nm: [] for nm in names}
    for pos, child in enumerate(names):
        cand = names[:pos]  # ensures acyclicity
        current = []
        best = local_bd_score(df, child, current, r)
        improved = True
        while improved and len(current) < 10:
            improved = False
            best_delta = 0.0
            best_p = None
            for p in cand:
                if p in current:
                    continue
                trial = sorted(current + [p])
                s = local_bd_score(df, child, trial, r)
                if s - best > best_delta:
                    best_delta = s - best
                    best_p = p
            if best_p is not None and best_delta > 0.0:
                current.append(best_p)
                best += best_delta
                improved = True
        parents[child] = list(current)
        for p in current:
            dag.add_edge(name2idx[p], name2idx[child])
    return dag, parents

def total_score(df, names, r, parents):
    return sum(local_bd_score(df, nm, parents[nm], r) for nm in names)
    

def compute(infile, outfile):
    df, names, r = read_csv(infile)
    dag, parents = k2(df, names, r)
    write_gph(dag, names, outfile)
    s = total_score(df, names, r, parents)
    print(f"Score: {s:.6f}")

def main():
    # if len(sys.argv) != 3:
    #     raise Exception("usage: python project1.py <infile>.csv <outfile>.gph")

    # inputfilename = sys.argv[1]
    # outputfilename = sys.argv[2]
    # compute(inputfilename, outputfilename)

    compute("data/large.csv", "large.gph")
    compute("data/medium.csv", "medium.gph")
    compute("data/small.csv", "small.gph")

if __name__ == '__main__':
    main()
