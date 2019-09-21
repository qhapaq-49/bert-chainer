import sentencepiece as spm
import sys
spm.SentencePieceTrainer.Train("--input=" + sys.argv[1] +" --model_prefix=m --add_dummy_prefix=false --vocab_size=2000 --character_coverage=0.9999999 --control_symbols=[PAD],[CLS],[SEP],[MASK]")
