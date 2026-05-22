GPU=$1
SPLIT_1=$2
SPLIT_2=$3
TASK=$4
MODEL=$5
if [ -z $6 ]; then
    EVAL_SAMPLE_RATIO=1
else
    EVAL_SAMPLE_RATIO=$6
fi
if [ -z $7 ] || [ -z $8 ]; then
    SEED_START=42
    SEED_END=46
else
    SEED_START=$7
    SEED_END=$8
fi

echo "GPU: $GPU"
echo "SPLIT_1: $SPLIT_1"
echo "SPLIT_2: $SPLIT_2"
echo "TASK: $TASK"
echo "MODEL: $MODEL"
echo "EVAL_SAMPLE_RATIO: $EVAL_SAMPLE_RATIO"
echo "SEED_START: $SEED_START"
echo "SEED_END: $SEED_END"

if [ $MODEL -eq 1 ]; then
    LOAD_MODE="FULL"
elif [ $MODEL -eq 2 ]; then
    LOAD_MODE="MIXED"
elif [ $MODEL -eq 3 ]; then
    LOAD_MODE="LIMITED"
fi

dataset=datasets/joint-v65
log_path=logs/transfer/$SPLIT_1-to-$SPLIT_2/$LOAD_MODE/$TASK
save_path=checkpoints/transfer/$SPLIT_1-to-$SPLIT_2/$LOAD_MODE/$TASK
output_path=output/transfer/$SPLIT_1-to-$SPLIT_2/$LOAD_MODE/$TASK
load_path=checkpoints/transfer/$SPLIT_1/$LOAD_MODE/best_checkpoint

if [ ! -d $output_path ]; then
    mkdir -p $output_path
fi

GPU_last=${GPU
GPU_num=$(($(echo $GPU | tr -cd ',' | wc -c) + 1))
EPOCH=200

for DOWN_NUM in 512 4096; do
    batchsize=256
    for SEED in $(seq $SEED_START $SEED_END); do
        CUDA_VISIBLE_DEVICES=$GPU accelerate launch --config_file hconfig.yaml --num_processes $GPU_num hmaintask_downsample_absolute_eval_sample.py $dataset $log_path $DOWN_NUM-$SEED --loadpath $load_path --savepath $save_path/$DOWN_NUM-$SEED --tasks $TASK --hop 2 --fanout 20 --maxepoch $EPOCH --patience 10 --eval_per_epoch 2 --batchsize $batchsize --lr 3e-4 --wd 2e-4 --num_mp 4 --use_rev True --use_gate False --fewshotfanout 3 --hiddim 512 --downsample_num $DOWN_NUM --downsample_seed $SEED --eval_sample_ratio $EVAL_SAMPLE_RATIO --eval_sample_seed $SEED &> $output_path/$DOWN_NUM-$SEED.log
    done
done

