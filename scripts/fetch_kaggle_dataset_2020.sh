TRAINING_DATA_URL="fjpa121197/imageclefmed2021"
SAMPLE_TRAINING_DATA_URL="fjpa121197/imageclef-sample"
NOW=$(date)

kaggle datasets download -d $SAMPLE_TRAINING_DATA_URL -p packages/cnn_modality_clf/cnn_modality_clf/datasets/ && \
unzip packages/cnn_modality_clf/cnn_modality_clf/datasets/imageclef-sample.zip -d packages/cnn_modality_clf/cnn_modality_clf/datasets/imageclef-sample && \
echo $SAMPLE_TRAINING_DATA_URL 'retrieved on:' $NOW > packages/cnn_modality_clf/cnn_modality_clf/datasets/training_data_reference.txt