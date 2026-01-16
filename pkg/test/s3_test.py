from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import random
import time

import transformers
from infini_gram.engine import InfiniGramEngine

def main():

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "allenai/OLMo-7B",
        add_bos_token=False,
        add_eos_token=False
    )

    engine = InfiniGramEngine(
        index_dir="",
        s3_names=['v4_dolmasample_olmo'],
        eos_token_id=tokenizer.eos_token_id,
        read_type="s3",
    )

    # Timing sequential.
    # times = []
    # for it in range(5):
    #     query_ids = [random.randint(0, 65535) for _ in range(5)]
    #     start_time = time.time()
    #     result = engine.count(input_ids=query_ids)
    #     print(result)
    #     end_time = time.time()
    #     times.append(end_time - start_time)
    # print('Average time:', np.mean(times))

    # Timing threading.
    start = time.time()
    with ThreadPoolExecutor(max_workers=50) as executor:
        futures = (
            executor.submit(
                engine.count,
                input_ids=query_ids
            )
            for query_ids in [tokenizer.encode("what has Donald Trump been up to lately") for _ in range(10)]
        )

        for future in as_completed(futures):
            result = future.result()
            print(result)
    end = time.time()
    print('Total time:', end - start)

    # input_ids = [5613, 4086, 9068]
    #
    # print(engine.count(input_ids=input_ids))
    # print()
    # print(engine.prob(prompt_ids=input_ids[:-1], cont_id=input_ids[-1]))
    # print()
    # print(engine.ntd(prompt_ids=input_ids[:-1]))
    # print()
    # print(engine.infgram_prob(prompt_ids=[3234, 4324] + input_ids[:-1], cont_id=input_ids[-1]))
    # print()
    # print(engine.infgram_ntd(prompt_ids=[3234, 4324] + input_ids[:-1]))
    # print()
    # print(engine.search_docs(input_ids=input_ids))
    # print()
    # find_result = engine.find(input_ids=input_ids)
    # rank = find_result['segment_by_shard'][0][0]
    # doc = engine.get_doc_by_rank_2(s=0, rank=rank, needle_len=len(input_ids), max_ctx_len=10)
    # print(doc)
    # print()


if __name__ == '__main__':
    main()