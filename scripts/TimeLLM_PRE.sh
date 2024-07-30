model_name=TimeLLM
train_epochs=40
learning_rate=0.005
llama_layers=4

master_port=20097
num_process=8
batch_size=1
d_model=16
d_ff=16

comment='TimeLLM-PRE'

accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/PRE/ \
  --data_path weather.csv \
  --model_id PRE \
  --model $model_name \
  --data CHLA \
  --features W \
  --seq_len 46 \
  --label_len 23 \
  --pred_len 46\
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --d_model 32 \
  --d_ff 32 \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment

