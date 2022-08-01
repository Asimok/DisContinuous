import argparse
import json
import logging
import os
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from transformers import AutoConfig, AdamW, get_linear_schedule_with_warmup

from dataLoader import load_dataset
from evaluate import evaluateAll
from getResults import perResults, disContinuousPredict
from QA_model import QuestionAnswering

logger = logging.getLogger(__name__)

import sys

sys.setrecursionlimit(10000)  # 防止Rouge计算深度爆炸


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args):
    logger.info("***** Prepare for Train *****")
    logger.info("Load dataset from file %s ...", os.path.join(args.datasetPath, args.trainFile))
    dataset, examples, _, tokenizer = load_dataset(args, evaluate=False)

    # import time
    # import pynvml
    #
    # while True:
    #     pynvml.nvmlInit()
    #     handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    #     meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    #     if meminfo.free / meminfo.total > 0.9:
    #         break
    #     # print(meminfo.free / meminfo.total)
    #     time.sleep(5)
    # time.sleep(10)

    logger.info("Load model from file %s ...", args.modelPath)
    config = AutoConfig.from_pretrained(args.modelPath)
    model = QuestionAnswering.from_pretrained(args.modelPath, from_tf=False, config=config)
    # model = QuestionAnswering.from_pretrained(r"/data2/wangbingchao/output/DisContinuous/BIO/epoch-2/", from_tf=False, config=config)
    model.to(args.device)

    args.batch_size = args.per_gpu_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(dataset)
    train_dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=args.batch_size)
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    logger.info("***** Running training *****")
    logger.info("Num examples = %d", len(dataset))
    logger.info("Num Epochs = %d", args.num_train_epochs)
    logger.info("Instantaneous batch size per GPU = %d", args.per_gpu_batch_size)
    logger.info("Total train batch size (w. parallel, distributed & accumulation) = %d", args.batch_size * args.gradient_accumulation_steps)
    logger.info("Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("Total optimization steps = %d", t_total)

    global_step = 1

    model.zero_grad()
    # Added here for reproductibility
    set_seed(args)

    # test(args, model, savedir="epoch-{}".format(0))

    for epoch in range(int(args.num_train_epochs)):
        epoch_iterator = tqdm(train_dataloader, desc="Epoch " + str(epoch))
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)

            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "BIOs": batch[3],
                "checks": batch[4],
            }

            loss, _ = model(**inputs)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            epoch_iterator.set_postfix(loss=loss.item())

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

            # Save model checkpoint
            if args.save_steps > 0 and global_step % args.save_steps == 0:
                savePath = os.path.join(args.savePath, "checkpoint-{}".format(global_step))
                # Take care of distributed/parallel training
                torch.save(args, os.path.join(savePath, "training_args.bin"))
                model_to_save = model.module if hasattr(model, "module") else model
                model_to_save.save_pretrained(savePath)
                tokenizer.save_pretrained(savePath)
                logger.info("Saving model checkpoint to %s", savePath)
                torch.save(optimizer.state_dict(), os.path.join(savePath, "optimizer.pt"))
                torch.save(scheduler.state_dict(), os.path.join(savePath, "scheduler.pt"))
                logger.info("Saving optimizer and scheduler states to %s", savePath)

        savePath = os.path.join(args.savePath, "epoch-{}".format(epoch))
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(savePath)
        tokenizer.save_pretrained(savePath)
        logger.info("Saving model checkpoint to %s", savePath)

        test(args, model, savedir="epoch-{}".format(epoch))


def test(args, model=None, savedir=""):
    logger.info("***** Prepare for Test *****")
    logger.info("Load dataset from file %s ...", args.testFile)
    dataset, examples, document_offsets, tokenizer = load_dataset(args, evaluate=True)
    args.batch_size = args.per_gpu_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.batch_size)

    if model is None:
        logger.info("Load model from file %s ...", args.modelPath)
        config = AutoConfig.from_pretrained(args.modelPath)
        model = QuestionAnswering.from_pretrained(args.modelPath, from_tf=False, config=config)
        model.to(args.device)
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

    logger.info("***** Running test *****")
    logger.info("Num examples = %d", len(dataset))
    logger.info("Batch size = %d", args.batch_size)

    selects = []

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }
            _, P_logits, C_logits = model(**inputs)

        token_type_ids = batch[2]
        for P_logit, C_logit, token_type_id in zip(P_logits, C_logits, token_type_ids):
            P_logit = P_logit.detach().cpu().tolist()
            C_logit = C_logit.detach().cpu().tolist()
            token_type_id = token_type_id.detach().cpu().tolist()
            question_len = 0
            for i, type_id in enumerate(token_type_id):
                if type_id == 1:
                    question_len = i
                    break
            passage_len = sum(token_type_id)

            select = perResults(P_logit, C_logit, question_len, passage_len)

            selects.append(select)
    logger.info("Evaluation done.")

    logger.info("Compute predictions.")
    messages, datas = examples
    predictions = {}
    for [qid, *_], [_, document, _], document_offset, select in zip(messages, datas, document_offsets, selects):
        predictions[qid] = disContinuousPredict(document, document_offset, select)

    logger.info("Save predictions.")
    output_prediction_file = os.path.join(args.savePath, savedir, "predictions.json")
    with open(output_prediction_file, 'w') as opf:
        json.dump(predictions, opf, ensure_ascii=False, indent=4)

    logger.info("Evaluate prediction.")
    results = evaluateAll(predictions, messages)
    logger.info(results)
    result_file = os.path.join(args.savePath, savedir, "results")
    with open(result_file, 'w') as rf:
        json.dump(results, rf, ensure_ascii=False, indent=4)
    logger.info("Save result.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='')
    parser.add_argument('--do_train', action="store_true")
    parser.add_argument('--do_test', action="store_true")
    parser.add_argument('--threePoint', action="store_true")
    parser.add_argument("--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--datasetPath', type=str, required=True)
    parser.add_argument('--trainFile', type=str)
    parser.add_argument('--testFile', type=str)
    parser.add_argument('--modelPath', type=str, required=True)
    parser.add_argument('--savePath', type=str, required=True)
    parser.add_argument('--tempPath', type=str, required=True)
    parser.add_argument("--per_gpu_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--threads", default=1, type=int, help="multiple threads for converting example to features")
    parser.add_argument("--num_train_epochs", default=4, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--learning_rate", default=3e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--save_steps", type=int, default=5000, help="Save checkpoint every X updates steps.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Number of updates steps to accumulate before performing a backward/update pass.",
                        )
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.",
                        )
    parser.add_argument("--max_query_length", default=64, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.",
                        )
    parser.add_argument("--n_best_size", default=20, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json output file.",
                        )
    parser.add_argument("--max_answer_length", default=500, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.",
                        )
    parser.add_argument("--seed", type=int, default=7455100, help="random seed for initialization")
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()

    set_seed(args)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
    )

    if not os.path.isdir(args.datasetPath):
        exit("datasetPath IS NOT A DIRCTIONARY. " + args.datasetPath)
    if not os.path.isdir(args.modelPath):
        exit("modelPath IS NOT A DIRCTIONARY. " + args.modelPath)
    if not os.path.isdir(args.tempPath):
        exit("tempPath IS NOT A DIRCTIONARY. " + args.tempPath)

    testFile = os.path.join(args.datasetPath, args.testFile)
    if not os.path.isfile(testFile):
        exit("There is no testFile OR testFile is not EXIST. " + testFile)

    if args.do_train:
        if not os.path.isfile(os.path.join(args.datasetPath, args.trainFile)):
            exit("There is no trainFile OR trainFile is not EXIST. " + os.path.join(args.datasetPath, args.trainFile))
        train(args)
    elif args.do_test:
        test(args)
