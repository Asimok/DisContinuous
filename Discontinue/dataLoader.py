import bisect
import json
import logging
import os

import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm
from transformers import AutoTokenizer, is_torch_available

logger = logging.getLogger(__name__)


def get_examples(datasetFile: str, evaluate=False):
    """
    :return:
         [messages, datas]
    messages -> [question_id, fakeAnswer, is_discontinue, answerPiece, part] \n
    datas -> [question, document, answerPosition]
    """
    if not datasetFile:
        logger.error("DatasetFile need a datasetFile to pass in")
    messages = []
    datas = []
    with open(datasetFile, 'r') as df:
        for line in df.readlines():
            jsonLine = json.loads(line)
            question_id = jsonLine["question_id"]
            document = jsonLine["document"]
            question = jsonLine["question"]
            fakeAnswer = jsonLine["fakeAnswer"]
            is_discontinue = None
            answerPiece = jsonLine["answerPiece"]
            answerPosition = jsonLine["answerPosition"]
            part = jsonLine["part"]
            # if len(document) + len(question) > 500:
            #     continue
            message = [question_id, fakeAnswer, is_discontinue, answerPiece, part]
            data = [question, document, answerPosition]
            messages.append(message)
            datas.append(data)
    examples = [messages, datas]
    return examples


def load_dataset(args, evaluate=False):
    """
    :return: dataset, examples, document_offsets, tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(args.modelPath, use_fast=True)
    temp_file = "{}_cache_{}_{}_{}".format(
        args.model_name,
        "dev" if evaluate else "train",
        args.testFile if evaluate else args.trainFile,
        args.modelPath.strip('/').split("/")[-1],
    )
    temp_file = os.path.join(args.tempPath, temp_file)
    if os.path.exists(temp_file) and not args.overwrite_cache:
        logger.info("Loading dataset from cached file %s", temp_file)
        dataset_and_examples = torch.load(temp_file)
        dataset, examples, offsets = dataset_and_examples["dataset"], dataset_and_examples["examples"], dataset_and_examples["offsets"]
        return dataset, examples, offsets, tokenizer
    else:
        logger.info("Creating dataset ...")

    if evaluate:
        examples = get_examples(os.path.join(args.datasetPath, args.testFile), evaluate=evaluate)
    else:
        examples = get_examples(os.path.join(args.datasetPath, args.trainFile), evaluate=evaluate)

    _, datas = examples

    document_offsets = []

    input_ids = []
    attention_masks = []
    token_type_ids = []
    BIOs = []
    checks = []

    for question, document, answerPosition in tqdm(datas, desc='Tokenizer'):
        text_token = tokenizer(document, add_special_tokens=False)
        document_id = text_token[0].ids
        document_offset = text_token[0].offsets
        # TODO: What it done under here may I delete it ?
        if evaluate:
            document_word = {}
            for i, w in enumerate(text_token[0].words):
                if w not in document_word:
                    document_word[w] = i
            document_offset = [(document_offset[document_word[w]][0], o[1]) for o, w in zip(document_offset, text_token[0].words)]
        # May delete end here.
        document_offsets.append(document_offset)
        question_id = tokenizer.encode(question, truncation=True, max_length=args.max_query_length)
        question_length = len(question_id)
        document_id = document_id[:args.max_seq_length - question_length - 1]
        document_length = len(document_id)
        input_id = question_id + document_id + [tokenizer.sep_token_id]
        attention_mask = [1] * len(input_id)
        token_type_id = [0] * question_length + [1] * (document_length + 1)
        for embedding in input_id, attention_mask, token_type_id:
            embedding += [tokenizer.pad_token_id] * (args.max_seq_length - len(embedding))
        BIO = [1] * args.max_seq_length  # 0:inner,1:out
        check = [-100] * args.max_seq_length  # 0:start,1:inner,2:end,-100:out
        for start, end in answerPosition:
            if start >= document_length:
                break
            if end > document_length:
                end = document_length
            start_pos = bisect.bisect_left(document_offset, (start, start))
            end_pos = bisect.bisect_right(document_offset, (end, end)) - 1
            for i in range(question_length + start_pos, question_length + end_pos + 1):
                BIO[i] = 0
                check[i] = 1
            check[question_length + start_pos] = 0
            check[question_length + end_pos] = 2

        input_ids.append(input_id)
        attention_masks.append(attention_mask)
        token_type_ids.append(token_type_id)
        BIOs.append(BIO)
        checks.append(check)

    if not is_torch_available():
        raise RuntimeError("PyTorch must be installed to return a PyTorch dataset.")

    logger.info("Created dataset length = %d.", len(input_ids))

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor(input_ids, dtype=torch.long)
    all_attention_masks = torch.tensor(attention_masks, dtype=torch.long)
    all_token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)
    all_BIOs = torch.tensor(BIOs, dtype=torch.long)
    all_checks = torch.tensor(checks, dtype=torch.long)

    if evaluate:
        dataset = TensorDataset(
            all_input_ids,
            all_attention_masks,
            all_token_type_ids,
        )
    else:
        dataset = TensorDataset(
            all_input_ids,
            all_attention_masks,
            all_token_type_ids,
            all_BIOs,
            all_checks,
        )

    logger.info("Saving dataset into cached file %s", temp_file)
    torch.save({"dataset": dataset, "examples": examples, "offsets": document_offsets}, temp_file)
    return dataset, examples, document_offsets, tokenizer
