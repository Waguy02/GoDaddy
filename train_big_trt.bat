echo m1. TRAINED python run_training_transformer.py -c -dv -de=64 -dce=1 -nl=4 -nh=4 -df=256 -sl=20 -bs=256 -do=0 -e200

echo python run_training_transformer.py -c -dv -de=128 -dce=2 -nl=6 -nh=8 -df=1024 -sl=20 -bs=512 -do=0.0001 -e400 -lr 5e-4

python run_training_transformer.py -c -dv -de=128 -dce=2 -nl=6 -nh=8 -df=1024 -sl=20 -bs=32 -do=0 -e40 -lr 5e-4