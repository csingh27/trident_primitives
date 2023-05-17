Train:  
# Primitives dataset (Replace primitives with wood-primitives and dark-primitives)
TRIDENT - 

N Way, K Shots
python -m src.trident_train --cnfg configs/primitives-N,K/train_conf.json --dataset primitives

MAML - 
python -m src.maml_train --cnfg configs/primitives-5,5/train_conf.json --dataset primitives

# Note: Change the path to the dataset when using WOOD-PRIMITIVES or DARK-PRIMITIVES 
# in the loaders file

# MiniImageNet dataset
python -m src.trident_train --cnfg configs/mini-5,5/train_conf.json

Test: 
Save the model in models/ folder
# Primitives dataset
python -m src.trident_test --cnfg configs/primitives-5,5/test_conf.json --dataset primitives
# MiniImageNet dataset
python -m src.trident_test --cnfg configs/mini-5,5/test_conf.json
