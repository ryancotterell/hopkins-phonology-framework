#!/bin/bash


train=/mounts/Users/cisintern/ryan/hopkins-phonology-framework/data/other/hebrew_splits2/hebrew_formatted2-train-1
dev=/mounts/Users/cisintern/ryan/hopkins-phonology-framework/data/other/hebrew_splits2/hebrew_formatted2-dev-1
test=/mounts/Users/cisintern/ryan/hopkins-phonology-framework/data/other/hebrew_splits2/hebrew_formatted2-test-1

#train=/mounts/Users/cisintern/ryan/hopkins-phonology-framework/data/other/egyptian2
#dev=/mounts/Users/cisintern/ryan/hopkins-phonology-framework/data/other/egyptian2
#test=/mounts/Users/cisintern/ryan/hopkins-phonology-framework/data/other/egyptian2

#train=/mounts/Users/cisintern/ryan/hopkins-phonology-framework/data/other/egyptian_splits/egyptian-train-1
#dev=/mounts/Users/cisintern/ryan/hopkins-phonology-framework/data/other/egyptian_splits/egyptian-dev-1
#test=/mounts/Users/cisintern/ryan/hopkins-phonology-framework/data/other/egyptian_splits/egyptian-test-1

train=/mounts/Users/cisintern/ryan/hopkins-phonology-framework/data/other/egyptian_splits2/egyptian2-train-$1
dev=/mounts/Users/cisintern/ryan/hopkins-phonology-framework/data/other/egyptian_splits2/egyptian2-dev-$1
test=/mounts/Users/cisintern/ryan/hopkins-phonology-framework/data/other/egyptian_splits2/egyptian2-test-$1



#train=$dev
#dev=$test
# HEBREW
python templatic/templatic_example.py $train $dev $test $2 $3 $4
# EGYPTIAN ARABIC
#python src/templatic_example.py data/other/egyptian2
