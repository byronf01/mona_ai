import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset 

num_proc = 8
num_proc_load_dataset = num_proc
enc = tiktoken.get_encoding("gpt2")

if __name__ == '__main__':
    
    dataset = load_dataset("stanfordnlp/SHP", num_proc=num_proc_load_dataset)

    
    """
    # make validation dataset 
    split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
    print(split_dataset)
    split_dataset['val'] = split_dataset.pop('test')    
    """
    dataset.pop('test')
    print(dataset)
    # tokenize dataset
    def process(example):
        question = enc.encode_ordinary(example['history']) 
        answer = enc.encode_ordinary(example['human_ref_A'])
        question.append(enc.eot_token) 
        answer.append(enc.eot_token) 
        out = {'question': question, 'answer': answer, 'len': len(question) + len(answer)}
        return out

    # tokenize the dataset
    tokenized = dataset.map(
        process,
        remove_columns=['post_id', 'domain', 'upvote_ratio', 'c_root_id_A', 'c_root_id_B', 
                        'created_at_utc_A', 'created_at_utc_B', 'score_A', 'score_B', 
                        'human_ref_B', 'labels', 'seconds_difference', 'score_ratio'],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    # concatenate all the ids in each dataset into one large file we can use for training
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
        dtype = np.uint16 
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()
