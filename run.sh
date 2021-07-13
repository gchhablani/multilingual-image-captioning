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
    --logging_steps 300 \
    --eval_steps 300 \
    --save_steps 1000 \
    --data_dir /home/user/data/CC12M/images \
    --train_file /home/bhavitvya_malik/final_data/train_file_batch.tsv \
    --validation_file /home/bhavitvya_malik/final_data/val_file_batch.tsv \
    --save_total_limit 2 \
    --push_to_hub \
    --num_train_epochs 2 \
    --push_to_hub_organization flax-community \
    --push_to_hub_token $token \
    --max_train_samples 100000 \
    --max_eval_samples_per_lang 25000 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 32 \