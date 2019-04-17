#export PRETRAINED_MODEL_PATH='model/model.ckpt-1400000'
export PRETRAINED_MODEL_PATH="/nfs/ai16storage01/sec/akp2/1706nasubi/inatomi/benchmark/ankoku/bert-chainer/data/pretraining_output/seq_128_model_snapshot_iter_$1.npz"
export FINETUNE_OUTPUT_DIR="model/livedoor_output_$1"
export BERT_BASE_DIR=/nfs/ai16storage01/sec/akp2/1706nasubi/inatomi/benchmark/bert-chainer/model

python3 run_classifier.py \
  --task_name=livedoor \
  --bert_config_file=/nfs/ai16storage01/sec/akp2/1706nasubi/inatomi/benchmark/bert-chainer/model/bert_config.json \
  --model_file=$BERT_BASE_DIR/wiki-ja.model \
  --vocab_file=$BERT_BASE_DIR/wiki-ja.vocab \
  --output_dir=model/livedoor_output \
  --gpu=$2 \
  --do_train=True \
  --do_eval=True \
  --data_dir=/nfs/ai16storage01/sec/akp2/1706nasubi/inatomi/benchmark/bert-chainer/data/livedoor \
  --do_lower_case=True \
  --init_checkpoint=$PRETRAINED_MODEL_PATH \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=2e-4 \
  --num_train_epochs=20 \
