export BERT_BASE_DIR=/nfs/ai16storage01/sec/akp2/1706nasubi/inatomi/benchmark/bert-chainer/model
# * wikipedia一つ17320
  # * 2709ファイル
  # * 46919880document
# * train_batch_size: 128
# * num_train_steps:366561
# batch_size128の
export checkpoint=/nfs/ai16storage01/sec/akp2/1706nasubi/inatomi/benchmark/bert-chainer/data/pretraining_output/model_snapshot_iter_27000.npz

# batchsizeが2倍なので学習率も2倍とする

# init_checkpointから学習を再開できるのか
python run_pretraining.py \
  --input_file=./data/wiki_data_pickle \
  --output_dir=./data/pretraining_output \
  --do_train=True \
  --do_eval=False \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --train_batch_size=128 \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --num_train_steps=365000 \
  --num_warmup_steps=10000 \
  --learning_rate=2e-4
