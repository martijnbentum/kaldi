#!/usr/bin/env bash
# Copyright 2012  Johns Hopkins University (author: Daniel Povey)
#           2021  Ke Li

# This script trains an RNN (LSTM and GRU) or Transformer-based
# language model with PyTorch and performs N-best and lattice rescoring.
# The N-best rescoring is in a batch computation mode as well.

# Dev/eval92 perplexities of the Transformer LM used for rescoring are: 55.7/71.1
# Baseline WER with a 4-gram LM:
# %WER 2.36 [ 133 / 5643, 10 ins, 11 del, 112 sub ] exp/chain/tdnn1g_sp/decode_bd_tgpr_eval92_fg//wer_13_1.0
# N-best rescoring:
# %WER 1.63 [ 92 / 5643, 7 ins, 5 del, 80 sub ] exp/chain/tdnn1g_sp/decode_bd_tgpr_eval92_fg_pytorch_transformer_nbest//wer_10_0.0
# Lattice rescoring:
# %WER 1.58 [ 89 / 5643, 6 ins, 6 del, 77 sub ] exp/chain/tdnn1g_sp/decode_bd_tgpr_eval92_fg_pytorch_transformer//wer_10_0.0

# Begin configuration section.
stage=1
exp=exp
decode_dir_suffix=pytorch_transformer
pytorch_path=/vol/tensusers/mbentum/FRISIAN_ASR/LM/rnn/pytorch
nn_model=$pytorch_path/fame_council

model_type=Transformer # LSTM, GRU or Transformer
embedding_dim=768
hidden_dim=768
nlayers=8
nhead=8
learning_rate=0.1
seq_len=100
dropout=0.2

oov='<UNK>' # Symbol for out-of-vocabulary words

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh
[ -z "$cmd" ] && cmd=$decode_cmd

set -e


#also used for testing (location of words.text; which maps words to integers)
data_dir=OLD/data/pytorchnn
#location to store
ac_model_dir=$exp/chain_dnn/tdnn1a_sp_bi

# Check if PyTorch is installed to use with python
if python3 steps/pytorchnn/check_py.py 2>/dev/null; then
  echo PyTorch is ready to use on the python side. This is good.
else
  echo PyTorch not found on the python side.
  echo Please install PyTorch first. For example, you can install it with conda:
  echo "conda install pytorch torchvision cudatoolkit=10.2 -c pytorch", or
  echo with pip: "pip install torch torchvision". If you already have PyTorch
  echo installed somewhere else, you need to add it to your PATH.
  echo Note: you need to install higher version than PyTorch 1.1 to train Transformer models
  exit 1
fi

# skip training - already done seperately


# training skipped

LM=bd_tgpr
if [ $stage -le 1 ]; then
  echo "$0: Perform N-best rescoring on $ac_model_dir with a $model_type LM. stage 1"
  #for decode_set in dev test; do
  for decode_set in $dev $dev'_nl' $dev'_fr' $dev'_mx' $test $test'_nl' $test'_fr' $test'_mx'; do
	echo "decoding $decode_set"
      decode_dir=${ac_model_dir}/decode_${decode_set}
      steps/pytorchnn/lmrescore_nbest_pytorchnn.sh \
        --cmd "$cmd --mem 4G" \
        --N 20 \
        --weight 0.7 \
        --model-type $model_type \
        --embedding_dim $embedding_dim \
        --hidden_dim $hidden_dim \
        --nlayers $nlayers \
        --nhead $nhead \
        --oov-symbol "'$oov'" \
        data/lang_test $nn_model $data_dir/words.txt \
        data/test_hires ${decode_dir} \
        ${decode_dir}_${decode_dir_suffix}_nbest
  done
fi


exit 0
