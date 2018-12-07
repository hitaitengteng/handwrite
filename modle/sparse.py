import numpy as np
import random
#转化一个序列列表为稀疏矩阵
def sparse_tuple_from(sequences, dtype=np.int32):
    """
    Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

    return indices, values, shape

# a=sparse_tuple_from([[1,2],[1,7],[1,2,5,10],[1,3,5,7,9,2,4]])




def decode_sparse_tensor(sparse_tensor):
    indexs=sparse_tensor[0]

    vales=sparse_tensor[1].tolist()
    all_index=[]
    all_label=[]

    for index in range(len(indexs.tolist())):
        all_index.append(indexs[index][0])
    xx=0
    xx1=0
    for i in list(set(all_index)):
        aa=all_index.count(i)
        xx += aa
        all_label.append(vales[xx1:xx])
        xx1+=aa
    return np.array(all_label)
# decode_sparse_tensor(a)











# decode_sparse_tensor(a)