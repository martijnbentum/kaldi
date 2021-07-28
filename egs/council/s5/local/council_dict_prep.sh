#!/usr/bin/env bash
# Copyright 2015-2016  Sarah Flora Juan
# Copyright 2016  Johns Hopkins University (Author: Yenda Trmal)
# Copyright 2016  Radboud University (Author: Emre Yilmaz)

# Apache 2.0

corpus=/vol/tensusers3/Frisiansubtitling/COUNCIL

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

if [ ! -d "$corpus" ] ; then
    echo >&2 "The directory $corpus does not exist"
fi

echo "creating the data/lang and data/local/dict directories"
mkdir -p data/lang data/local/dict


cat $corpus/lexicon.txt > data/local/dict/lexicon.txt
echo "!SIL	SIL" >> data/local/dict/lexicon.txt
echo "<UNK>	SPN" >> data/local/dict/lexicon.txt
env LC_ALL=C sort -u -o data/local/dict/lexicon.txt data/local/dict/lexicon.txt
cat data/local/dict/lexicon.txt | \
    perl -ane 'print join("\n", @F[1..$#F]) . "\n"; '  | \
    sort -u | grep -v 'SIL' > data/local/dict/nonsilence_phones.txt


touch data/local/dict/extra_questions.txt
touch data/local/dict/optional_silence.txt

echo "SIL"   > data/local/dict/optional_silence.txt
echo "SIL"   > data/local/dict/silence_phones.txt
echo "<UNK>" > data/local/dict/oov.txt

echo "Dictionary preparation succeeded"
