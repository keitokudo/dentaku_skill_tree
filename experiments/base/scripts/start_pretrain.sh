set -eu
if [ $# = 2 ]; then
    config_file_path=$1
    gpu_id=$2
else
    echo "Select config file path and gpu id"
    exit
fi

zsh ./scripts/all_prepro.sh
zsh ./scripts/train.sh $config_file_path $gpu_id

