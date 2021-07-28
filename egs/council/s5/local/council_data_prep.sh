#!/usr/bin/env bash
# Copyright 2015-2016  Sarah Flora Juan
# Copyright 2016  Johns Hopkins University (Author: Yenda Trmal)
# Copyright 2016  Radboud University (Author: Emre Yilmaz)
# Copyright 2021  Radboud University (Author: Martijn Bentum)

# Apache 2.0

corpus=/vol/tensusers3/Frisiansubtitling/COUNCIL
data_dir='language_split_and_fame'

if [ $data_dir == 'simple' ] ; then
	directories=(train dev test)
elif [ $data_dir == 'language_split' ] ; then
	directories=(train dev test dev_nl dev_fr dev_mx test_nl test_fr test_mx)
elif [ $data_dir == 'language_split_and_fame' ] ; then
	directories=(train dev test dev_nl dev_fr dev_mx test_nl test_fr test_mx)
	directories+=(fame_dev_nl fame_dev_fr fame_dev_mx fame_test_nl fame_test_fr fame_test_mx)
fi

. utils/parse_options.sh

if [ ! -d "$corpus" ] ; then
    echo >&2 "The directory $corpus does not exist"
fi

echo "Preparing train, development and test data for the following directories:"
echo ${directories[@]}
echo "----"

mkdir -p data data/local 
for x in ${directories[@]}; do
	mkdir -p data/$x
done

for x in ${directories[@]}; do
    echo "Copy spk2utt, utt2spk, wav.scp, text for $x"
    cp $corpus/data/$x/text     data/$x/text    || exit 1;
    cp $corpus/data/$x/utt2spk  data/$x/utt2spk || exit 1;
    cp $corpus/data/$x/segments  data/$x/segments || exit 1;
    cp $corpus/data/$x/wav.scp  data/$x/wav.scp || exit 1;


    # fix_data_dir.sh fixes common mistakes (unsorted entries in wav.scp,
    # duplicate entries and so on). Also, it regenerates the spk2utt from
    # utt2sp
    utils/fix_data_dir.sh data/$x 
	utils/validate_data_dir.sh data/$x --no-feats
    printf "done with $x\n\n"
done

echo "copying the LM arpa file"
gzip -c $corpus/FAME_council_mix_50.lm > data/local/LM.gz

echo "Data preparation completed."

