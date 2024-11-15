#!/bin/bash

num_adapters=0
exp_id="a"
batch_size=6

dtype="float32"
model="llama7b"

nlayers=32

python exp.py --exp_id $exp_id --nlayers $nlayers --dtype $dtype \
--model $model --method offmate_no_act_offload --batch_size $batch_size

python exp.py --exp_id $exp_id --nlayers $nlayers --dtype $dtype \
--model $model --method offmate --batch_size $batch_size

python exp.py --exp_id $exp_id --nlayers $nlayers --dtype $dtype \
--model $model --method offmate_no_cpu_optim --batch_size $batch_size

python exp.py --exp_id $exp_id --nlayers $nlayers --dtype $dtype \
--model $model --method offmate_base --batch_size $batch_size

for nlayers in 4 3 2 1
do 
    python exp.py --exp_id $exp_id --nlayers $nlayers --dtype $dtype \
    --model $model --method torch --batch_size $batch_size
done