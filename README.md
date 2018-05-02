rnnlm2wfst
==========

https://github.com/glecorve/rnnlm2wfst

Conversion of recurrent neural network language models to weighted finite state transducers
This directory contains all the code to run the conversion

Gwénolé Lecorvé
Idiap Research Institute, Martigny, Switzerland
2011-2012

Prerequisites
-------------

OpenFst: You can use your own version. In that case, edit the makefile in rnnlm-0.2b/src.
BLAS: Better if you have it installed (much faster).

Configuration & compilation
---------------------------

Same as install.sh.

### git clone https://github.com/glecorve/rnnlm2wfst.git

### OpenFst
	将openfst 1.6.3版本下载，并解压
	cd openfst-1.6.3
	./configure --prefix=`pwd`
	make install
	cd ..
	
### K-means
	cd kmeans
	make
	cd ..
	
### RNNLM
	修改src/makefile：
	1）将CC = g++ 修改为 CC = g++ -std=c++11
	2）将OPENFST:=../../openfst-1.3.2/ 修改为OPENFST:=../../openfst-1.6.3
	如果出现error while loading shared libraries: libXXX.so.X: cannot open shared object file: No such file错误的话，需要修改下列文件
	1）cd /etc/ld.so.conf.d/
	2）vi fst.conf
	3）添加之前编译好的openfst lib路径，比如/home/pjs/rnn2wfst/rnnlm2wfst/openfst-1.6.3/lib
	4）ldconfig
	
	cd rnnlm-0.2b
	# Do not use USE_BLAS=1 if BLAS is not installed
	make USE_BLAS=1
	cd ..

Examples
--------

### RNNLM basic usage
	See examples/example_rrnlm.sh

### Train RNNLM
	bin/rnnlm -train examples/rnn2wfst.train.txt -valid examples/rnn2wfst.dev.txt -rnnlm examples/rnn2wfst.model -hidden 2 -rand-seed 1 -bptt 3 -debug 2 -class 1

### Write logs of continuous states
	bin/trace-hidden-layer -rnnlm examples/rnn2wfst.model -text examples/rnn2wfst.train.txt >  examples/rnn2wfst.train.trace
	
### Generate artificial data (if you think training data is too small or anything else)
	bin/rnnlm -rnnlm examples/rnn2wfst.model -gen 10000 | tail -n +2 > examples/rnn2wfst.generated.txt
	bin/trace-hidden-layer -rnnlm examples/rnn2wfst.model -text examples/rnn2wfst.generated.txt >  examples/rnn2wfst.generated.trace
	
### Build K-means (flat or hiearchical)
	perl bin/build-cluster-hierarchy.pl examples/rnn2wfst.train.trace 2 4 > examples/rnn2wfst.4.kmeans
	perl bin/build-cluster-hierarchy.pl examples/rnn2wfst.train.trace 2 1 8 > examples/rnn2wfst.1+8.kmeans
	perl bin/build-cluster-hierarchy.pl examples/rnn2wfst.train.trace 2 1 2 4 8 > examples/rnn2wfst.1+2+4+8.kmeans

### Cluster-based convertion
	time bin/rnn2fst -rnnlm examples/rnn2wfst.model -fst examples/rnn2wfst.k1+8.p1e-3.fst -discretize examples/rnn2wfst.1+8.kmeans -hcluster -prune 1e-3 -backoff 2
	
	Remark: the value of the backoff option (2) is the depth of the cluster hieararchy.
	
### See the resulting WFST
	fstprint examples/rnn2wfst.k1+8.p1e-3.fst

### Simulate perplexity with discretized RNNLM but without pruning
	bin/rnnlm -rnnlm examples/rnn2wfst.model -test examples/rnn2wfst.test.txt -debug 2 -discretize examples/rnn2wfst.1+8.kmeans | less

### Measure perplexity
	bin/wfst-ppl -fst examples/rnn2wfst.k1+8.p1e-3.fst -text examples/rnn2wfst.test.txt | less
	
### Describe WFST
	../openfst-1.3.2/bin/fstinfo --info_type=long examples/rnn2wfst.k1+8.p1e-3.fst
