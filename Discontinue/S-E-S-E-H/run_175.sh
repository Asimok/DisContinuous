export CUDA_VISIBLE_DEVICES=5,6,7
python run.py \
  --model_name SEH \
  --do_train \
  --do_test \
  --save_steps -1 \
  --num_train_epochs 11 \
  --warmup_steps 500 \
  --per_gpu_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --datasetPath /data0/wangbingchao/temp/DisContinuous/dataset/3.0remakeDiscontinueDataset/ \
  --trainFile train.json \
  --testFile dev.json \
  --modelPath /data2/wangbingchao/database/bert_pretrained/bert-base-chinese \
  --savePath /data2/maqi/output/out2 \
  --tempPath /data2/maqi/temp/temp