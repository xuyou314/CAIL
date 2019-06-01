#!/usr/bin/env bash
./word2vec -train /home/xuyou/CAIL/wiki.zh.simple.seg.txt -output \
/home/xuyou/CAIL/w2v_veciter10.txt -size 200 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 1 -iter 10
