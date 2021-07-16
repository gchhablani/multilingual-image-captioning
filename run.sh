read -p "Enter repo name: " repo

if [ -d $repo ]
then
    cd $repo
    git pull
    cd ..
else
    git clone https://huggingface.co/flax-community/$repo
fi

token=$(cat /home/bhavitvya_malik/.huggingface/token)

echo "Token found $token"
python3 main.py \
    --output_dir $repo \
    --seed 42 \
    --logging_steps 5000 \
    --eval_steps 10000 \
    --save_steps 15000 \
    --data_dir /home/user/data/CC12M/images \
    --train_file /home/bhavitvya_malik/final_data/train_file_batch.tsv \
    --validation_file /home/bhavitvya_malik/final_data/val_file_batch.tsv \
    --save_total_limit 10 \
    --push_to_hub \
    --num_train_epochs 3 \
    --push_to_hub_organization flax-community \
    --push_to_hub_token $token \
    --max_eval_samples_per_lang 25000 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 64 \
    --preprocessing_num_workers 8 \
    --warmup_steps 10000 \
    --learning_rate 1e-4 \
    --predict_with_generate False \