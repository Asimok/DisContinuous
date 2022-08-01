export CUDA_VISIBLE_DEVICES=0
python run.py \
  --model_name SEH_Dis \
  --do_train \
  --do_test \
  --save_steps -1 \
  --num_train_epochs 10 \
  --warmup_steps 500 \
  --per_gpu_batch_size 32 \
  --gradient_accumulation_steps 1 \
  --datasetPath /data1/wangbingchao/dataset/DureaderToDis/allDis/ \
  --trainFile train.json \
  --testFile test.json \
  --modelPath /data0/wangbingchao/pretrained/torch/bert-base-chinese \
  --savePath /data1/wangbingchao/output/DureaderToDis/allDis \
  --tempPath /data1/wangbingchao/temp
