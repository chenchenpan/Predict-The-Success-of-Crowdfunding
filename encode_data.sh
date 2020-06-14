DATA_DIR=/Users/cicipan/projects/Predict-The-Success-of-Crowdfunding/raw_data_webrobots
DATA_NAME=kickstarter
CONFIG_FILE=config.json
USE_TEXT_FEATURES=True


python encode_data.py --data_dir $DATA_DIR \
					  --data_name $DATA_NAME \
                      --config_file $CONFIG_FILE \
                      --use_text_features $USE_TEXT_FEATURES

