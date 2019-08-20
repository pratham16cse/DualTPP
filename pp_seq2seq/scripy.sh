#!/bin/bash

alg_name=$1
dataset_name=$2
alg_path=$3
dataset_path=$4
constraints=${5:-default}
export CUDA_VISIBLE_DEVICES=""
batch_size=64

# Exit the script if dataset_path does not exist
if [ ! -d $dataset_path ]; then  echo >&2 "$dataset_path does not exist"; exit 1; fi

last_commit_id="$(git log --format="%h" -n 1)"
echo $last_commit_id

stashed=0
if [ -z "$(git status --porcelain --untracked-files=no)" ]; then
    echo "No local changes"
else
    echo "Local changes exist, a stash is needed"
    git stash
    stashed=1
fi

# ----- Create output directory ----- #
if [ -d "Outputs/Experiment_"$last_commit_id ]; then
    if [ -d "Outputs/Experiment_"$last_commit_id"/"$dataset_name ]; then
        if [ -d "Outputs/Experiment_"$last_commit_id"/"$dataset_name"/"$constraints ]; then
	    if [ -d "Outputs/Experiment_"$last_commit_id"/"$dataset_name"/"$constraints"/"$alg_name ]; then
            echo "Outputs/Experiment_"$last_commit_id"/"$dataset_name"/"$constraints"/"$alg_name "already exists." 1>&2
#            echo "Exiting . . ." 1>&2
#	    if [ $stashed==1 ]; then
#                git stash pop
#            fi
#            exit 1
             else
                 mkdir "Outputs/Experiment_"$last_commit_id"/"$dataset_name"/"$constraints"/"$alg_name
	     fi
        else
            mkdir "Outputs/Experiment_"$last_commit_id"/"$dataset_name"/"$constraints
            mkdir "Outputs/Experiment_"$last_commit_id"/"$dataset_name"/"$constraints"/"$alg_name
        fi
    else
        mkdir "Outputs/Experiment_"$last_commit_id"/"$dataset_name
        mkdir "Outputs/Experiment_"$last_commit_id"/"$dataset_name"/"$constraints
        mkdir "Outputs/Experiment_"$last_commit_id"/"$dataset_name"/"$constraints"/"$alg_name
    fi
else
    mkdir "Outputs/Experiment_"$last_commit_id
    mkdir "Outputs/Experiment_"$last_commit_id"/"$dataset_name
    mkdir "Outputs/Experiment_"$last_commit_id"/"$dataset_name"/"$constraints
    mkdir "Outputs/Experiment_"$last_commit_id"/"$dataset_name"/"$constraints"/"$alg_name
fi
output_dir="Outputs/Experiment_"$last_commit_id"/"$dataset_name"/"$constraints"/"$alg_name
# ----- #

# ----- Create save_dir and summary_dir ----- #
save_dir=$output_dir"/saved_model"
summary_dir=$output_dir"/summary"
if [ ! -d $save_dir ]; then
    mkdir $save_dir
fi
if [ ! -d $summary_dir ]; then
    mkdir $summary_dir
fi
# ----- #

print_dump=$output_dir"/print_dump"

command="python3.6 \
	 $alg_path/src/run.py \
	 $dataset_name $alg_name \
	 $dataset_path"/train.event" \
	 $dataset_path"/train.time" \
	 $dataset_path"/dev.event" \
	 $dataset_path"/dev.time" \
	 $dataset_path"/test.event" \
	 $dataset_path"/test.time" \
	 --dataset_path $dataset_path\
	 --save $save_dir \
	 --cpu-only \
	 --epochs 50 \
	 --init-learning-rate 0.001 \
	 --patience 0 \
	 --stop-criteria per_epoch_val_err \
	 --epsilon 0.1 \
	 --normalization average_per_seq
	 --init-zero-dec-state \
         --no-concat-final-enc-state \
	 >>$print_dump"

#if [[ "$dataset_name" == *"data_bookorder"* ]]; then
#	command=$command" --normalization average"
#fi

if [ "$constraints" != "default" ]; then
	command=$command" --constraints "$constraints
fi

eval $command

#python3.6 \
#	$alg_path/src/run.py \
#	$dataset_name $alg_name \
#	$dataset_path"/train.event" \
#	$dataset_path"/train.time" \
#	$dataset_path"/dev.event" \
#	$dataset_path"/dev.time" \
#	$dataset_path"/test.event" \
#	$dataset_path"/test.time" \
#	--save $save_dir \
#	--cpu-only \
#	--epochs 100 \
#	--init-learning-rate 0.001 \
#	>>$print_dump

if [ $stashed==1 ]; then
    git stash pop
fi
