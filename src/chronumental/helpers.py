import functools
import jax
import jax.numpy as jnp


def get_unnnamed_node_label(i):
    name = f"NODE_{i:07d}"
    return name

def preorder_traversal(node):
    yield node
    for clade in node.children:
        yield from preorder_traversal(clade)
        
# Credit: Guillem Cucurull http://gcucurull.github.io/deep-learning/2020/06/03/jax-sparse-matrix-multiplication/
@functools.partial(jax.jit, static_argnums=(2))
def sp_matmul(A, B, shape):
    """
    Arguments:
        A: (N, M) sparse matrix represented as a tuple (indexes, values)
        B: (M,K) dense matrix
        shape: value of N
    Returns:
        (N, K) dense matrix
    """
    # In theory this performs an unnecessary multiplication by 1,
    # (unnecessary for our purposes)
    # but it probably gets removed in the XLA compilation step.
    # Nevertheless we should ultimately refactor this.
    assert B.ndim == 2
    indexes, values = A
    rows, cols = indexes
    in_ = B.take(cols, axis=0)
    prod = in_ * values[:, None]
    res = jax.ops.segment_sum(prod, rows, shape)
    return res


def do_branch_matmul(rows, cols, branch_lengths_array, final_size):
    A = ((rows, cols), jnp.ones_like(cols))
    B = branch_lengths_array.reshape((branch_lengths_array.shape[0], 1))
    calc_dates = sp_matmul(A, B, final_size).squeeze()
    return calc_dates


def get_classic_intervals(tree,threshold=1e-9):
    time={}
    for node in tree.root.traverse_preorder():
        if node.is_root():
            time[node] = 0.
        else:
            time[node] = time[node.parent]
            if node.edge_length is not None:
                time[node] += node.edge_length
    origin = max(time.values())
    events = sorted([(origin-time[node],node) for node in time])
    time=0
    intervals=[]
    lineages=1
    lastTime = events[0][0]
    lastNode = events[0][1]
    nodeMap={}
    nodeQ =[events[0][1]]
    tips = [events[0][1]]
    for i in range(1,len(events)):
        #check if the events are super close together. if they are merge with last interval
        if events[i][0]-lastTime<threshold:
            if events[i][1].is_leaf():
                tips.append(events[i][1])
                lineages+=1
            else:
                lineages-=1
            nodeQ.append(events[i][1])
        else:      
            #closing and interval so store it and prep for next one
            interval = {"time":lastTime,"duration":events[i][0]-lastTime,"lineages":lineages,"index":len(intervals)}
            intervals.append(interval)

            for n in nodeQ:
                nodeMap[n]=interval

            if events[i][1].is_leaf():
                tips.append(events[i][1])
                lineages+=1
            else:
                lineages-=1
            lastTime = events[i][0]
            nodeQ = [events[i][1]]


    # root also maps to last interval
    nodeMap[tree.root]=intervals[len(intervals)-1]
    return (intervals,nodeMap,tips)

def get_cks(tips):
    sample_to_coalesce = {}
    visited = set()

    node = tips[0]
    while node.parent:
        node = node.parent
        visited.add(node)

    for i in range(1,len(tips)):
        node = tips[i]
        while node.parent not in visited:
            node = node.parent
            visited.add(node)
        sample_to_coalesce[tips[i]] = node.parent # coalesces at first previously visited node

    return sample_to_coalesce


def get_intervals(tree):
    rtt={}
    tips = []
    for node in tree.root.traverse_preorder():
        if node.is_root():
            rtt[node] = 0.
        else:
            rtt[node] = rtt[node.parent]
            if node.edge_length is not None:
                rtt[node] += node.edge_length
        if node.is_leaf():
            tips.append(node)
    tips = sorted(tips,key=lambda t: rtt[t])

    s_to_c = get_cks(tips)
 

    sk=[]
    ck=[]
    for i in range(1,len(tips)):
        tip=tips[i]
        cutoff = rtt[s_to_c[tip]]
        visited=set()

        done = False
        nextTip = i-1

        Ck=0
        while not done:
            node=tips[nextTip]
            while node is not s_to_c[tip] and node not in visited and rtt[node]>cutoff and not node.is_root():
                parent = node.parent
                if rtt[parent]<cutoff:
                    Ck+=rtt[node]-cutoff
                else:
                    Ck+=node.edge_length
                visited.add(node)
                node=parent
            nextTip -=1
            if nextTip==-1 or rtt[tips[nextTip]]<cutoff:
                done=True
        sk.append(rtt[tip])
        ck.append(Ck)
    return (jnp.array(sk),jnp.array(ck),s_to_c,tips)