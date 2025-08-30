python extract_dac_latents.py \
    --root_path /data/dataset \
    --file_list /data/learnable/speech/files.txt \
    --output_dir /data/dataset/metadata \
    --checkpoint ./ckpts/300k_20250829_044827/checkpoint.pt\
    --config ./configs/configx2.yml \
    --num_gpus 4 \
    --num_decode_samples 10


python extract_dac_latents.py \
    --root_path data_test \
    --output_dir data_test/metadata \
    --checkpoint ./checkpoint.pt \
    --config ./config.yml \
    --num_gpus 1 \
    --num_decode_samples 10