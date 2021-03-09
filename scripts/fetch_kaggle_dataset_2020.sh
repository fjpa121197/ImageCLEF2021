TRAINING_DATA_URL="fjpa121197/imageclef-med-concept-detection"
NOW=$(date)

kaggle datasets download -d $TRAINING_DATA_URL -p packages/cnn_modality_clf/cnn_modality_clf/datasets/