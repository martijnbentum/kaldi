#!/usr/bin/env bash

. ./cmd.sh # sets up run.pl or queue configuration (run.pl is without using a queue)
. ./path.sh # setes kaldi root and sorting to C sorting
. ./utils/parse_options.sh # helper script to parse command line argements

stage=4 			# sets the stage of processing
stop_stage=7 		# sets stage to stop processing, if 2 the program will nog execut stage 2
do_decoding=false 	#whether to do the decoding 
feat_nj=10 			#number of jobs for different stages (perhaps only important for queue processing)
train_nj=10
decode_nj=10
#famecorpus=/vol/tensusers/mbentum/FRISIAN_ASR/corpus

#OBSOLETE
#if [ -d $famecorpus ] ; then
#  echo "Fame corpus present. OK."
#elif [ -f ./fame.tar.gz ] ; then
#  echo "Unpacking..."
#  tar xzf fame.tar.gz
#elif [ ! -d $famecorpus ] && [ ! -f ./fame.tar.gz ] ; then
#  echo "The Fame! corpus is not present. Please register here: http://www.ru.nl/clst/datasets/ "
#  echo " and download the corpus and put it at $famecorpus" && exit 1
#fi

#probably settings for different training stages, untouched
numLeavesTri1=5000
numGaussTri1=25000
numLeavesMLLT=5000
numGaussMLLT=25000
numLeavesSAT=5000
numGaussSAT=25000
numGaussUBM=800
numLeavesSGMM=10000
numGaussSGMM=20000

if [ $stage -le 1 ]; then
  printf "\n\n--- stage one starting --- data preparation\n\n"
  #local/fame_data_prep.sh $famecorpus || exit 1;
  #local/fame_dict_prep.sh $famecorpus || exit 1;
  for x in train dev test; do
	printf "running fix_data_dir for $x\n"
	utils/fix_data_dir.sh data/$x
	printf "running validate_data_dir for $x\n"
	utils/validate_data_dir.sh data/$x
	printf "## $x done ##\n\n"
  done
  printf 'running prepare lang\n\n'
  utils/prepare_lang.sh data/local/dict "<UNK>" data/local/lang data/lang || exit 1;
  printf '\n\nrunning format_lm\n\n'
  utils/format_lm.sh data/lang data/local/FAME_council_mix_50.lm.gz data/local/dict/lexicon.txt data/lang_test || exit 1;
  printf "\n--- stage one done --- data preparation\n\n"
fi

if [ $stage -le 2 -a $stop_stage -gt 2 ]; then
  # Feature extraction
  printf "\n\n--- stage two starting --- feature extraction\n\n"
  for x in train dev test; do
      printf "running make_mfcc for $x\n"
      steps/make_mfcc.sh --nj $feat_nj --cmd "$train_cmd" data/$x exp/make_mfcc/$x mfcc || exit 1;
      printf "running compute_cmvn_stats for $x\n"
      steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x mfcc || exit 1;
	  printf "## $x done ##\n\n"
  done
  printf "\n--- stage two done --- feature extraction\n\n"
fi

if [ $stage -le 3 -a $stop_stage -gt 3 ]; then
  ### Monophone
  echo "Starting monophone training."
  steps/train_mono.sh --nj $train_nj --cmd "$train_cmd" data/train data/lang exp/mono || exit 1;
  echo "Mono training done."

  echo "Decoding the development and test sets using monophone models."
  utils/mkgraph.sh --mono data/lang_test exp/mono exp/mono/graph || exit 1;
  steps/decode.sh --nj $decode_nj --cmd "$decode_cmd" exp/mono/graph data/dev exp/mono/decode_dev || exit 1;
  steps/decode.sh --nj $decode_nj --cmd "$decode_cmd" exp/mono/graph data/test exp/mono/decode_test || exit 1;
  echo "Monophone decoding done."
fi


if [ $stage -le 4 -a $stop_stage -gt 4 ]; then
  ### Triphone
  echo "Starting triphone training."
  steps/align_si.sh --nj $train_nj --cmd "$train_cmd" data/train data/lang exp/mono exp/mono_ali || exit 1;
  steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd"  $numLeavesTri1 $numGaussTri1 data/train data/lang exp/mono_ali exp/tri1 || exit 1;
  echo "Triphone training done."

  echo "Decoding the development and test sets using triphone models."
  utils/mkgraph.sh data/lang_test exp/tri1 exp/tri1/graph || exit 1;
  steps/decode.sh --nj $decode_nj --cmd "$decode_cmd" exp/tri1/graph data/dev exp/tri1/decode_dev || exit 1;
  steps/decode.sh --nj $decode_nj --cmd "$decode_cmd" exp/tri1/graph data/test exp/tri1/decode_test || exit 1;
  echo "Triphone decoding done."
fi

if [ $stage -le 5 -a $stop_stage -gt 5 ]; then
  ### Triphone + LDA and MLLT
  echo "Starting LDA+MLLT training."
  steps/align_si.sh  --nj $train_nj --cmd "$train_cmd"  data/train data/lang exp/tri1 exp/tri1_ali || exit 1;
  steps/train_lda_mllt.sh  --cmd "$train_cmd"  --splice-opts "--left-context=3 --right-context=3" $numLeavesMLLT $numGaussMLLT data/train data/lang  exp/tri1_ali exp/tri2 || exit 1;
  echo "LDA+MLLT training done."

  echo "Decoding the development and test sets using LDA+MLLT models."
  utils/mkgraph.sh data/lang_test  exp/tri2 exp/tri2/graph || exit 1;
  steps/decode.sh --nj $decode_nj --cmd "$decode_cmd" exp/tri2/graph data/dev exp/tri2/decode_dev || exit 1;
  steps/decode.sh --nj $decode_nj --cmd "$decode_cmd" exp/tri2/graph data/test exp/tri2/decode_test || exit 1;
  echo "LDA+MLLT decoding done."
fi


if [ $stage -le 6 -a $stop_stage -gt 6 ]; then
  ### Triphone + LDA and MLLT + SAT and FMLLR
  echo "Starting SAT+FMLLR training."
  steps/align_si.sh  --nj $train_nj --cmd "$train_cmd" --use-graphs true data/train data/lang exp/tri2 exp/tri2_ali || exit 1;
  steps/train_sat.sh --cmd "$train_cmd" $numLeavesSAT $numGaussSAT data/train data/lang exp/tri2_ali exp/tri3 || exit 1;
  echo "SAT+FMLLR training done."

  echo "Decoding the development and test sets using SAT+FMLLR models."
  utils/mkgraph.sh data/lang_test exp/tri3 exp/tri3/graph || exit 1;
  steps/decode_fmllr.sh --nj $decode_nj --cmd "$decode_cmd" exp/tri3/graph data/dev exp/tri3/decode_dev || exit 1;
  steps/decode_fmllr.sh --nj $decode_nj --cmd "$decode_cmd" exp/tri3/graph data/test exp/tri3/decode_test || exit 1;
  echo "SAT+FMLLR decoding done."
fi


if [ $stage -le 7 -a $stop_stage -gt 7 ]; then
  echo "Starting SGMM training."
  steps/align_fmllr.sh --nj $train_nj --cmd "$train_cmd" data/train data/lang exp/tri3 exp/tri3_ali || exit 1;
  steps/train_ubm.sh --cmd "$train_cmd" $numGaussUBM data/train data/lang exp/tri3_ali exp/ubm || exit 1;
  steps/train_sgmm2.sh --cmd "$train_cmd" $numLeavesSGMM $numGaussSGMM data/train data/lang exp/tri3_ali exp/ubm/final.ubm exp/sgmm2 || exit 1;
  echo "SGMM training done."

  echo "Decoding the development and test sets using SGMM models"
  utils/mkgraph.sh data/lang_test exp/sgmm2 exp/sgmm2/graph || exit 1;
  steps/decode_sgmm2.sh --nj $decode_nj --cmd "$decode_cmd" --transform-dir exp/tri3/decode_dev exp/sgmm2/graph data/dev exp/sgmm2/decode_dev || exit 1;
  steps/decode_sgmm2.sh --nj $decode_nj --cmd "$decode_cmd" --transform-dir exp/tri3/decode_test exp/sgmm2/graph data/test exp/sgmm2/decode_test || exit 1;
  echo "SGMM decoding done."
fi

if [ $stage -le 8 -a $stop_stage -gt 8 ]; then
  echo "Starting DNN training and decoding."
  local/nnet/run_dnn.sh || exit 1;
  echo "start second step"
  local/nnet/run_dnn_fbank.sh || exit 1;
  echo "done DNN training"
fi

if  $do_decoding ; then
	echo "start decoding"
	#score
	for x in exp/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done
	echo "done decoding"
fi
