#!/bin/bash
echo "++++++++ ivector_common.sh ++++++++"
echo `date`

set -e -o pipefail


# This script is called from local/nnet3/run_tdnn.sh and local/chain/run_tdnn.sh (and may eventually
# be called by more scripts).  It contains the common feature preparation and iVector-related parts
# of the script.  See those scripts for examples of usage.

# Repurposed for CGN by LvdW 2017
# Repurposed for council by Martijn 2021 
stage=0
nj=30
min_seg_len=1.55  # min length in seconds... we do this because chain training
                  # will discard segments shorter than 1.5 seconds.   Must remain in sync
                  # with the same option given to prepare_lores_feats_and_alignments.sh
train_set='train'   # you might set this to e.g. train.
gmm='tri3'          # This specifies a GMM-dir from the features of the type you're training the system on;
                         # it should contain alignments for 'train_set'.
dev='dev'
test='test'
exp='exp'

num_threads_ubm=32
nnet3_affix=_dnn  # affix for exp/nnet3 directory to put iVector stuff in, so it
                         # becomes exp/nnet3_cleaned or whatever.

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh


gmm_dir=$exp/${gmm}
ali_dir=$exp/${gmm}

echo "running script with stage $stage"

echo "checking the existence of final.mdl files"
for f in data/${train_set}/feats.scp ${gmm_dir}/final.mdl; do
  if [ ! -f $f ]; then
    echo "$0: expected file $f to exist"
    exit 1
  fi
done

for datadir in ${train_set}_sp $dev $dev'_nl' $dev'_fr' $dev'_mx' $test $test'_nl' $test'_fr' $test'_mx' ; do
	echo $datadir
done


if [ $stage -le 2 ] && [ -f data/${train_set}_sp_hires/feats.scp ]; then
  echo "checks for stage 2"
  echo "$0: data/${train_set}_sp_hires/feats.scp already exists. "
  echo " ... Please either remove it, or rerun this script with stage > 2."
  exit 1
fi


if [ $stage -le 1 ]; then
  echo "$0: preparing directory for speed-perturbed data stage 1"
  utils/data/perturb_data_dir_speed_3way.sh data/${train_set} data/${train_set}_sp
fi

if [ $stage -le 2 ]; then
  echo "$0: creating high-resolution MFCC features, stage 2"
  
  #for datadir in ${train_set}_sp dev test; do
  for datadir in ${train_set}_sp $dev $dev'_nl' $dev'_fr' $dev'_mx' $test $test'_nl' $test'_fr' $test'_mx' ; do
    utils/copy_data_dir.sh data/$datadir data/${datadir}_hires
  done
  #cp data/dev/text_ref data/dev_hires/ #text_ref is identical to text, text_ref does not exist in my dev dir
  #cp data/test/text_ref data/test_hires/ #text_ref is identical to text, text_ref does not exist in my test dir

  # do volume-perturbation on the training data prior to extracting hires
  # features; this helps make trained nnets more invariant to test data volume.
  utils/data/perturb_data_dir_volume.sh data/${train_set}_sp_hires

  #for datadir in ${train_set}_sp dev test; do
  for datadir in ${train_set}_sp $dev $dev'_nl' $dev'_fr' $dev'_mx' $test $test'_nl' $test'_fr' $test'_mx' ; do
    steps/make_mfcc.sh --nj $nj --mfcc-config conf/mfcc_hires.conf \
      --cmd "$train_cmd" data/${datadir}_hires
    steps/compute_cmvn_stats.sh data/${datadir}_hires
    utils/fix_data_dir.sh data/${datadir}_hires
  done
fi

if [ $stage -le 3 ]; then
  echo "$0: combining short segments of speed-perturbed high-resolution MFCC training data | stage 3"
  # we have to combine short segments or we won't be able to train chain models
  # on those segments.
  utils/data/combine_short_segments.sh \
     data/${train_set}_sp_hires $min_seg_len data/${train_set}_sp_hires_comb

  echo "copying cmvn from original directories (train/dev/test) to the newly created sp_hires_com"
  # just copy over the CMVN to avoid having to recompute it.
  cp data/${train_set}_sp_hires/cmvn.scp data/${train_set}_sp_hires_comb/
  utils/fix_data_dir.sh data/${train_set}_sp_hires_comb/
fi

if [ $stage -le 4 ]; then
  echo "$0: selecting segments of hires training data that were also present in the"
  echo " ... original training data | stage 4 "

  # note, these data-dirs are temporary; we put them in a sub-directory
  # of the place where we'll make the alignments.
  temp_data_root=$exp/nnet3${nnet3_affix}/tri5
  mkdir -p $temp_data_root

  utils/data/subset_data_dir.sh --utt-list data/${train_set}/feats.scp \
    data/${train_set}_sp_hires $temp_data_root/${train_set}_hires

  # note: essentially all the original segments should be in the hires data.
  n1=$(wc -l <data/${train_set}/feats.scp)
  n2=$(wc -l <$temp_data_root/${train_set}_hires/feats.scp)
  if [ $n1 != $n1 ]; then
    echo "$0: warning: number of feats $n1 != $n2, if these are very different it could be bad."
  fi

  echo "$0: training a system on the hires data for its LDA+MLLT transform, in order to produce the diagonal GMM."
  if [ -e $exp/nnet3${nnet3_affix}/tri5/final.mdl ]; then
    # we don't want to overwrite old stuff, ask the user to delete it.
    echo "$0: $exp/nnet3${nnet3_affix}/tri5/final.mdl already exists: "
    echo " ... please delete and then rerun, or use a later --stage option."
    exit 1;
  fi

  steps/train_lda_mllt.sh --cmd "$train_cmd" --num-iters 7 --mllt-iters "2 4 6" \
    --splice-opts "--left-context=3 --right-context=3" \
    3000 10000 $temp_data_root/${train_set}_hires data/lang \
    $gmm_dir $exp/nnet3${nnet3_affix}/tri5
fi

if [ $stage -le 5 ]; then
  echo "$0: computing a subset of data to train the diagonal UBM | stage 5"

  temp_data_root=$exp/nnet3${nnet3_affix}/diag_ubm
  mkdir -p $temp_data_root

  # train a diagonal UBM using a subset of about a quarter of the data
  # we don't use the _comb data for this as there is no need for compatibility with
  # the alignments, and using the non-combined data is more efficient for I/O
  # (no messing about with piped commands).
  num_utts_total=$(wc -l <data/${train_set}_sp_hires/utt2spk)
  num_utts=$[$num_utts_total/4]
  utils/data/subset_data_dir.sh data/${train_set}_sp_hires \
    $num_utts ${temp_data_root}/${train_set}_sp_hires_subset

  echo "$0: training the diagonal UBM."
  # Use 512 Gaussians in the UBM.
  steps/online/nnet2/train_diag_ubm.sh --cmd "$train_cmd" --nj 30 \
    --num-frames 700000 \
    --num-threads $num_threads_ubm \
    ${temp_data_root}/${train_set}_sp_hires_subset 512 \
    $exp/nnet3${nnet3_affix}/tri5 $exp/nnet3${nnet3_affix}/diag_ubm
fi

if [ $stage -le 6 ]; then
  # Train the iVector extractor.  Use all of the speed-perturbed data since iVector extractors
  # can be sensitive to the amount of data.  The script defaults to an iVector dimension of
  # 100.
  echo "$0: training the iVector extractor | stage 6"
  steps/online/nnet2/train_ivector_extractor.sh --cmd "$train_cmd" --nj 10 \
    data/${train_set}_sp_hires $exp/nnet3${nnet3_affix}/diag_ubm $exp/nnet3${nnet3_affix}/extractor || exit 1;
fi

if [ $stage -le 7 ]; then
  echo "extracting ivector ?? | stage 7"
  # note, we don't encode the 'max2' in the name of the ivectordir even though
  # that's the data we extract the ivectors from, as it's still going to be
  # valid for the non-'max2' data, the utterance list is the same.
  ivectordir=$exp/nnet3${nnet3_affix}/ivectors_${train_set}_sp_hires_comb
  
  # We extract iVectors on the speed-perturbed training data after combining
  # short segments, which will be what we train the system on.  With
  # --utts-per-spk-max 2, the script pairs the utterances into twos, and treats
  # each of these pairs as one speaker; this gives more diversity in iVectors..
  # Note that these are extracted 'online'.

  # having a larger number of speakers is helpful for generalization, and to
  # handle per-utterance decoding well (iVector starts at zero).
  temp_data_root=${ivectordir}
  utils/data/modify_speaker_info.sh --utts-per-spk-max 2 \
    data/${train_set}_sp_hires_comb ${temp_data_root}/${train_set}_sp_hires_comb_max2

  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj $nj \
    ${temp_data_root}/${train_set}_sp_hires_comb_max2 \
    $exp/nnet3${nnet3_affix}/extractor $ivectordir

  # Also extract iVectors for the test data, but in this case we don't need the speed
  # perturbation (sp) or small-segment concatenation (comb).
  #for data in dev test; do
  for data in $dev $dev'_nl' $dev'_fr' $dev'_mx' $test $test'_nl' $test'_fr' $test'_mx' ; do
    steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 4 \
      data/${data}_hires $exp/nnet3${nnet3_affix}/extractor \
      $exp/nnet3${nnet3_affix}/ivectors_${data}_hires
  done
fi

if [ -f data/${train_set}_sp/feats.scp ] && [ $stage -le 9 ]; then
  echo "$0: $feats already exists.  Refusing to overwrite the features "
  echo " to avoid wasting time.  Please remove the file and continue if you really mean this."
  exit 1;
fi


if [ $stage -le 8 ]; then
  echo "stage 8"
  for x in s t; do
    echo "$0: preparing directory for low-resolution speed-perturbed data (for alignment)"
    utils/data/perturb_data_dir_speed_3way.sh \
      data/${train_set} data/${train_set}_sp
  done
fi

if [ $stage -le 9 ]; then
  echo "$0: making MFCC features for low-resolution speed-perturbed data | stage 9"

  steps/make_mfcc.sh --nj $nj \
    --cmd "$train_cmd" data/${train_set}_sp
  steps/compute_cmvn_stats.sh data/${train_set}_sp
  echo "$0: fixing input data-dir to remove nonexistent features, in case some "
  echo ".. speed-perturbed segments were too short."
  utils/fix_data_dir.sh data/${train_set}_sp
fi

if [ $stage -le 10 ]; then
  echo "$0: combining short segments of low-resolution speed-perturbed  MFCC data | stage 10"
  src=data/${train_set}_sp
  dest=data/${train_set}_sp_comb
  utils/data/combine_short_segments.sh $src $min_seg_len $dest
  # re-use the CMVN stats from the source directory, since it seems to be slow to
  # re-compute them after concatenating short segments.
  cp $src/cmvn.scp $dest/
  utils/fix_data_dir.sh $dest
fi

if [ $stage -le 11 ]; then
  echo "perfoming check for stage 11"
  if [ -f ${ali_dir}_train_sp_comb/ali.1.gz ]; then
    echo "$0: alignments in ${ali_dir}_dnn appear to already exist.  Please either remove them "
    echo " ... or use a later --stage option."
    exit 1
  fi
  echo "$0: aligning with the perturbed, short-segment-combined low-resolution data | stage 11"
  steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
    data/${train_set}_sp_comb data/lang $gmm_dir ${ali_dir}_train_sp_comb
fi

echo "done with all steps"

exit 0;
