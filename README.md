# Predict-The-Success-of-Crowdfunding
A Neural Network model can predict the success of crowdfunding projects

This project is using the Kickstarter (latest 2019) and Indiegogo (latest 2017) datasets collected from [Webrobots.io](https://webrobots.io/projects/).

This example code can encode kickstarter dataset, which contains numerical, categorical, and textual features, and use TF-IDF encode the textual features.

```shell
export DATA_DIR=$HOME/projects/Predict-The-Success-of-Crowdfunding/raw_data_webrobots
export DATA_NAME=kickstarter
export CONFIG_FILE=config.json
export USE_TEXT_FEATURES=True
export ENCODE_TEXT_WITH=tfidf

python encode_data_separate.py --data_dir $DATA_DIR \
                               --data_name $DATA_NAME \
                               --config_file $CONFIG_FILE \
                               --use_text_features $USE_TEXT_FEATURES \
                               --encode_text_with $ENCODE_TEXT_WITH
```
