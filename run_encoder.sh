export DATA_DIR=$HOME/projects/Predict-The-Success-of-Crowdfunding/raw_data_webrobots
export GLOVE_DIR=$HOME/projects/Predict-The-Success-of-Crowdfunding/glove
export DATA_NAME=KICK
export OUTPUT_DIR=$HOME/projects/Predict-The-Success-of-Crowdfunding/outputs_test


python encoder.py --data_dir $DATA_DIR \
				  --data_name $DATA_NAME \
				  --output_dir $OUTPUT_DIR\
				  --metadata_file metadata_comb.json \
				  --use_text_features True \
				  --encode_text_with glove\
				  --glove_dir $GLOVE_DIR \
				  --glove_file glove.6B.50d.txt \
				  --max_words 20\
				  --max_sequence_length 5\
				  --embedding_dim 50




