import random
from pcfg import PCFG
import threading

def generate_probs(num_options):
    if num_options <= 0:
        raise ValueError("Number of options must be positive")

    # Generate random integers for each option
    random_ints = [random.randint(1, 100) for _ in range(num_options)]

    # Calculate the total sum
    total = sum(random_ints)

    # Normalize each integer by the total sum to get probabilities
    probs = [i / total for i in random_ints]

    return probs


def create_random_pcfg(
    num_nonterminals,
    num_terminals,
    rhs_max_options=5,
    rhs_max_len=5,
    constrain_to_pfsa=False,
):
    # Create non-terminal symbols
    nonterminals = [f"N{i}" for i in range(num_nonterminals)]

    # Create terminal symbols as consecutive integers
    terminals = [f"'{i}'" for i in range(num_terminals)]

    # Initialize production rules
    productions = []

    for lhs in nonterminals:
        rhs_options_ct = random.randint(1, rhs_max_options)
        rhs_option_probs = generate_probs(rhs_options_ct)

        rhs_options = []

        for rhs_option_prob in rhs_option_probs:
            rhs = []

            if constrain_to_pfsa:
                rhs.append(
                    random.choice(nonterminals + terminals)
                )  # TODO: is this the right constraint?
            else:
                # Randomly decide the length of the right-hand side (at least 1)
                rhs_len = random.randint(1, rhs_max_len)
                for _ in range(rhs_len):
                    rhs.append(random.choice(nonterminals + terminals))

            rhs_option = f"{' '.join(rhs)} [{rhs_option_prob}]"
            rhs_options.append(rhs_option)

        production = f"{lhs} -> {' | '.join(rhs_options)}"
        productions.append(production)

    start_production = f"S -> {' | '.join([f'{nonterminal} [{1/len(nonterminals)}]' for nonterminal in nonterminals])}"
    productions.insert(0, start_production)

    # Create the PCFG
    grammar = PCFG.fromstring("\n".join(productions))

    return grammar


def generate_dataset(
    num_nonterminals,
    num_terminals,
    rhs_max_options,
    rhs_max_len,
    constrain_to_pfsa,
    num_toks_total,
    num_toks_per_seq=256,
) -> list[str]:
    print(num_nonterminals, num_terminals, rhs_max_options, rhs_max_len)

    grammar = create_random_pcfg(
        num_nonterminals,
        num_terminals,
        rhs_max_options=rhs_max_options,
        rhs_max_len=rhs_max_len,
        constrain_to_pfsa=constrain_to_pfsa,
    )

    dataset = []
    total_tokens_generated = 0

    while total_tokens_generated < num_toks_total:
        document_tokens = 0
        document = []

        while document_tokens < num_toks_per_seq:
            try:
                sentence = next(grammar.generate(1))
            except RecursionError:
                continue
            except StopIteration:
                print('No more sentences to generate')
                break  # No more sentences can be generated

            sentence_token_count = sentence.count(" ") + 2

            available_space = num_toks_per_seq - document_tokens
            if sentence_token_count <= available_space:
                document.append(sentence)
                document_tokens += sentence_token_count
            else:
                # Split the sentence into words and add words until the document is full
                words = sentence.split()
                words_to_add = words[:available_space]
                truncated_sentence = " ".join(words_to_add)

                document.append(truncated_sentence)
                document_tokens += len(words_to_add)

            if document_tokens == num_toks_per_seq:
                break

        if document:
            dataset.append(" 0 ".join(document))
            total_tokens_generated += document_tokens

        if total_tokens_generated >= num_toks_total or not document:
            break  # Stop if we've met the total token count or can't generate more documents

    return dataset

def generate_dataset_part(grammar, num_toks_per_seq, target_tokens, dataset, total_tokens_generated, lock):
    local_dataset = []
    local_tokens_generated = 0
    while local_tokens_generated < target_tokens:
        document_tokens = 0
        document = []
        while document_tokens < num_toks_per_seq:
            try:
                sentence = next(grammar.generate(1))
            except RecursionError:
                continue
            except StopIteration:
                print('No more sentences to generate')
                break
            
            print(sentence)
            sentence_token_count = sentence.count(" ") + 2
            available_space = num_toks_per_seq - document_tokens
            if sentence_token_count <= available_space:
                document.append(sentence)
                document_tokens += sentence_token_count
            else:
                words = sentence.split()
                words_to_add = words[:available_space]
                truncated_sentence = " ".join(words_to_add)
                document.append(truncated_sentence)
                document_tokens += len(words_to_add)

            if document_tokens == num_toks_per_seq:
                break

        if document:
            local_dataset.append(" 0 ".join(document))
            local_tokens_generated += document_tokens

        if local_tokens_generated >= target_tokens or not document:
            break

    with lock:
        dataset.extend(local_dataset)
        total_tokens_generated[0] += local_tokens_generated

def generate_dataset_threaded(
    num_nonterminals,
    num_terminals,
    rhs_max_options,
    rhs_max_len,
    constrain_to_pfsa,
    num_toks_total,
    num_toks_per_seq=256,
) -> list[str]:
    # NOTE: threaded dataset generation isn't noticeably faster
    print(num_nonterminals, num_terminals, rhs_max_options, rhs_max_len)

    num_threads = 16
    threads = []
    lock = threading.Lock()
    dataset = []
    total_tokens_generated = 0
    target_tokens_per_thread = num_toks_total // num_threads

    grammar = create_random_pcfg(
        num_nonterminals,
        num_terminals,
        rhs_max_options=rhs_max_options,
        rhs_max_len=rhs_max_len,
        constrain_to_pfsa=constrain_to_pfsa,
    )

    for _ in range(num_threads):
        thread = threading.Thread(target=generate_dataset_part, args=(grammar, num_toks_per_seq, target_tokens_per_thread, dataset, total_tokens_generated, lock))
        threads.append(thread)
        thread.start()
    
    print(dataset)

    for thread in threads:
        thread.join()

    return dataset

if __name__ == "__main__":
    from data_utils import (
        calculate_median_stdev_gzipability,
        count_total_tokens,
        pcfg_dataset_to_dataloader,
        upload_to_huggingface,
    )
    from transformers import AutoTokenizer

    context_length = 256
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-2-7b-chat-hf", token="[REDACTED]"
    )
    tokenizer.add_special_tokens({"pad_token": "<pad>"})

    dataset_stats = [
        # (3, 20, 2, 2, False),
        # (10, 150, 5, 3, False),
        # (20, 300, 10, 5, False),
        # (30, 400, 10, 8, False),
        # (100, 2000, 100, 30, False),
        # (50, 500, 20, 15, False),

        # (10, 600, 5, 10, False), # .32
        # (20, 300, 15, 5, False), # .36
        # (30, 200, 10, 15, False), # .38
        # (50, 100, 20, 20, False), # .34

        # (3, 300, 2, 2, False), # isovocab
        # (10, 300, 5, 3, False),
        # (20, 300, 10, 5, False),
        # (50, 300, 20, 10, False),
        (100, 300, 100, 30, False),
        (200, 300, 200, 50, False),
    ]
    for row in dataset_stats: # NOTE: runs one dataset generation + upload at a time
        dataset_stats = [row]

        pcfg_datasets = [
            generate_dataset(*row, 100_000_000, num_toks_per_seq=context_length)
            for row in dataset_stats
        ]
        med_std_gzips = [
            calculate_median_stdev_gzipability(pcfg_dataset)
            for pcfg_dataset in pcfg_datasets
        ]
        for i, pcfg_dataset in enumerate(pcfg_datasets):
            med, std = med_std_gzips[i]
            total_toks = count_total_tokens(
                pcfg_dataset_to_dataloader(pcfg_dataset, padder_tokenizer=tokenizer)
            )

            print(
                f"{i}: {med:.3f} +- {std:.3f} ({total_toks})  | [{' '.join([str(x) for x in dataset_stats[i]])}]"
            )
            upload_to_huggingface(pcfg_dataset, med, dataset_stats[i])
