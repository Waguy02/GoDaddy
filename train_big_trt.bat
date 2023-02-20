echo python run_training_transformer.py -c -dv -de=64 -dce=1 -nl=6 -nh=4 -df=256 -sl=20 -bs=16 -do=0 -e250 -lr 5e-4

echo python run_training_transformer.py -c -dv -de=128 -dce=3 -nl=8 -nh=8 -df=256 -sl=15 -bs=32 -do=0 -e700 -lr 5e-4

python run_training_transformer.py -c -dv -de=128 -dce=3 -nl=8 -nh=8 -df=1024 -sl=18 -bs=512 -do=0.1 -e2000 -lr 1e-4

python run_training_transformer.py -c -dv -de=128 -dce=3 -nl=8 -nh=8 -df=2048 -sl=18 -bs=16 -do=0. -e400 -lr 5e-4