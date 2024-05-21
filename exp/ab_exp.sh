#!/bin/bash

num_adapters=0
exp_id="a"
batch_size=4

model="phi2-3b"
dtype="float32"
nlayers=32
python exp.py --exp_id $exp_id --nlayers $nlayers --dtype $dtype \
--model $model --method offmate --batch_size $batch_size

python exp.py --exp_id $exp_id --nlayers $nlayers --dtype $dtype \
--model $model --method offmate_no_act_offload

python exp.py --exp_id $exp_id --nlayers $nlayers --dtype $dtype \
--model $model --method offmate_no_cpu_optim

for nlayers in 1 2 3 5 6 7 8
do 
    python exp.py --exp_id $exp_id --nlayers $nlayers --method torch --dtype $dtype --model $model
done

dtype="float32"
model="llama7b"

for nlayers in 32
python exp.py --exp_id $exp_id --nlayers $nlayers --dtype $dtype \
--model $model --method offmate --batch_size $batch_size

python exp.py --exp_id $exp_id --nlayers $nlayers --dtype $dtype \
--model $model --method offmate_no_act_offload

python exp.py --exp_id $exp_id --nlayers $nlayers --dtype $dtype \
--model $model --method offmate_no_cpu_optim

for nlayers in 4 3 2 1
do 
    python exp.py --exp_id $exp_id --nlayers $nlayers --method torch --dtype $dtype --model $model
done


model="llama13b"
dtype="bfloat16"


nlayers=40
python exp.py --exp_id $exp_id --nlayers $nlayers --dtype $dtype \
--model $model --method offmate

python exp.py --exp_id $exp_id --nlayers $nlayers --dtype $dtype \
--model $model --method offmate_no_act_offload

python exp.py --exp_id $exp_id --nlayers $nlayers --dtype $dtype \
--model $model --method offmate_no_cpu_optim

for nlayers in 4 3 2 1
do 
    python exp.py --exp_id $exp_id --nlayers $nlayers --method torch --dtype $dtype --model $model
done


model="phi2-2b"
dtype="float32"
nlayers=24
python exp.py --exp_id $exp_id --nlayers $nlayers --dtype $dtype \
--model $model --method offmate --batch_size $batch_size

python exp.py --exp_id $exp_id --nlayers $nlayers --dtype $dtype \
--model $model --method offmate_no_act_offload

python exp.py --exp_id $exp_id --nlayers $nlayers --dtype $dtype \
--model $model --method offmate_no_cpu_optim

for nlayers in 1 2 3 5 6 7 8
do 
    python exp.py --exp_id $exp_id --nlayers $nlayers --method torch --dtype $dtype --model $model
done



