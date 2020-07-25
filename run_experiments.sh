export ENCODED_DATA_DIR=$HOME/projects/Predict-The-Success-of-Crowdfunding/encoded_data
export SEARCH_SPACE=$HOME/projects/Predict-The-Success-of-Crowdfunding/raw_data_webrobots/KICK/search_space.json
export DATA_NAME=KICK
export OUTPUT_DIR=$HOME/projects/Predict-The-Success-of-Crowdfunding/outputs



python experiments.py --encoded_data_dir $ENCODED_DATA_DIR \
					  --data_name $DATA_NAME \
					  --search_space_filepath $SEARCH_SPACE \
					  --output_dir $OUTPUT_DIR \
					  --task_type classification \
					  --num_classes 2 \
					  --model_type mlp \
					  --num_trials 3
