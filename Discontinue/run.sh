export CUDA_VISIBLE_DEVICES=0
python run.py \
  --model_name our \
  --do_train \
  --do_test \
  --save_steps -1 \
  --num_train_epochs 10 \
  --warmup_steps 500 \
  --per_gpu_batch_size 32 \
  --gradient_accumulation_steps 1 \
  --datasetPath /Users/maqi/dataset/default/ \
  --trainFile train.json \
  --testFile dev.json \
  --modelPath bert-base-chinese \
  --savePath /Users/maqi/dataset/output/ \
  --tempPath /Users/maqi/dataset/temp
