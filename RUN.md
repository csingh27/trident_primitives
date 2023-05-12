# Primitives dataset
python -m src.trident_train --cnfg configs/primitives-5,5/train_conf.json --dataset primitives
python -m src.trident_test --cnfg configs/primitives-5,5/test_conf.json --dataset primitives
python -m src.maml_train --cnfg configs/primitives-5,5/train_conf.json --dataset primitives
# MiniImageNet dataset
python -m src.trident_train --cnfg configs/mini-5,1/train_conf.json
