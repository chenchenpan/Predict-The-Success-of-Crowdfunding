# Predict-The-Success-of-Crowdfunding
A Neural Network model can predict the success of crowdfunding projects

This project is using the Kickstarter (latest 2019) and Indiegogo (latest 2017) datasets collected from [Webrobots.io](https://webrobots.io/projects/).

This example code can encode kickstarter dataset, which contains numerical, categorical, and textual features, and it's using pre-trained GloVe to encode the textual feature.

```shell
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
                  --glove_file $GLOVE_DIR/glove.6B.50d.txt \
                  --max_words 20\
                  --max_sequence_length 5\
                  --embedding_dim 50
```
