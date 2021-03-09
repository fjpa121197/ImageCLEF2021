TRAINING_DATA_URL="fjpa121197/imageclef-med-concept-detection"
NOW=$(date)

kaggle datasets download -d $TRAINING_DATA_URL -p packages/cnn_modality_clf/cnn_modality_clf/datasets/ && \
unzip packages/cnn_modality_clf/cnn_modality_clf/datasets/imageclef-med-concept-detection.zip -d packages/cnn_modality_clf/cnn_modality_clf/datasets/imageclef-med-concept-detection && \
echo $TRAINING_DATA_URL 'retrieved on:' $NOW > packages/cnn_modality_clf/cnn_modality_clf/datasets/training_data_reference.txt && \