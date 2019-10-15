wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
tar -xzf  simple-examples.tgz
rm simple-examples.tgz

mkdir data
mv simple-examples/data/ptb.train.txt data/
mv simple-examples/data/ptb.valid.txt data/
mv simple-examples/data/ptb.test.txt data/
rm -r simple-examples
