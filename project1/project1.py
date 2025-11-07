import networkx as nx
import pandas as pd
import math
import matplotlib.pyplot as plt

from scipy.special import gammaln
import time

def write_gph(dag, idx2names, filename):
    with open(filename, 'w') as f:
        for edge in dag.edges():
            f.write("{}, {}\n".format(idx2names[edge[0]], idx2names[edge[1]]))

def read_csv(path):
    df = pd.read_csv(path)
    names = list(df.columns)
    for c in names:
        df[c] = df[c].astype(int)
    max_val = {c: int(df[c].max()) for c in names}
    return df, names, max_val

# Local Bayesian score
def score(df, child, parents, r):
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
    G = nx.DiGraph()
    G.add_nodes_from(range(len(names)))

    parents = {nm: [] for nm in names}
    name_to_index_map = {nm: i for i, nm in enumerate(names)}

    y = total_score(df, names, r, parents)

    for k in range(1, len(names)):
        i_name = names[k]
        y_cur = y

        while True:
            y_best = -math.inf
            j_best = None

            # try each prev node
            for j_name in names[:k]:
                if j_name in parents[i_name]:
                    continue

                G.add_edge(name_to_index_map[j_name], name_to_index_map[i_name])
                parents[i_name].append(j_name)

                y_new = total_score(df, names, r, parents)

                if y_new > y_best:
                    y_best = y_new
                    j_best = j_name

                parents[i_name].remove(j_name)
                G.remove_edge(name_to_index_map[j_name], name_to_index_map[i_name])

            if y_best > y_cur:
                y_cur = y_best
                y = y_best
                G.add_edge(name_to_index_map[j_best], name_to_index_map[i_name])
                parents[i_name].append(j_best)
            else:
                break

    return G, parents


def total_score(df, names, r, parents):
    return sum(score(df, nm, parents[nm], r) for nm in names)
    

def compute(infile, outfile):
    df, names, r = read_csv(infile)

    start_time = time.time()
    dag, parents = k2(df, names, r)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Doing stuff took {elapsed_time} seconds")

    write_gph(dag, names, outfile)
    s = total_score(df, names, r, parents)
    print(f"Score: {s:.6f}")

    plt.figure(figsize=(6, 4))
    labels = {i: name for i, name in enumerate(names)}
    nx.draw(dag, labels=labels, with_labels=True)
    plt.show()


def main():
    compute("data/large.csv", "large.gph")
    # compute("data/medium.csv", "medium.gph")
    # compute("data/small.csv", "small.gph")


if __name__ == '__main__':
    main()
