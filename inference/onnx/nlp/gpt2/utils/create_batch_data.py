from transformers import GPT2Tokenizer, GPT2LMHeadModel, DefaultDataCollator
from datasets import load_dataset
from itertools import chain

# datasets Preprocess #
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
def tokenize_function(examples):
    return tokenizer(examples["text"])

block_size = 1024

# Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

# DataLoaders creation
def prepare_datasets(dataset_dir):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", cache_dir=dataset_dir)

    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=None,
        remove_columns=["text"],
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )

    lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=None,
            load_from_cache_file=True,
            desc=f"Grouping texts in chunks of {block_size}",
        )
    return lm_datasets