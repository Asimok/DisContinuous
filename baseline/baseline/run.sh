export CUDA_VISIBLE_DEVICES=0,1,2,3
python run.py \
  --model_name baseline \
  --do_train \
  --do_test \
  --save_steps -1 \
  --num_train_epochs 4 \
  --warmup_steps 500 \
  --per_gpu_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --version_2_with_negative \
  --datasetPath /data0/wangbingchao/temp/DisContinuous/dataset/2.0deleteNoAnswerAndUnionDataset/ \
  --trainFile train.json \
  --testFile dev.json \
  --modelPath /data2/wangbingchao/database/bert_pretrained/bert-base-chinese \
  --savePath /data2/wangbingchao/output/DisContinuous/baseline \
  --tempPath /data2/wangbingchao/temp
