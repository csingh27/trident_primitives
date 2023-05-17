Train:  
# Primitives dataset (Replace primitives with wood-primitives and dark-primitives)
TRIDENT - 

10 Way, 5 Shots
python -m src.trident_train --cnfg configs/primitives-10,5/train_conf.json --dataset primitives

10 Way, 1 Shot
python -m src.trident_train --cnfg configs/primitives-10,1/train_conf.json --dataset primitives

5 Way, 5 Shots
python -m src.trident_train --cnfg configs/primitives-5,5/train_conf.json --dataset primitives

5 Way, 1 Shot
python -m src.trident_train --cnfg configs/primitives-5,1/train_conf.json --dataset primitives

MAML - 
python -m src.maml_train --cnfg configs/primitives-5,5/train_conf.json --dataset primitives

# Note: Change the path to the dataset when using WOOD-PRIMITIVES or DARK-PRIMITIVES 
in the loaders file

# MiniImageNet dataset
python -m src.trident_train --cnfg configs/mini-5,5/train_conf.json

Test: 
Save the model in models/ folder
# Primitives dataset
python -m src.trident_test --cnfg configs/primitives-5,5/test_conf.json --dataset primitives
# MiniImageNet dataset
python -m src.trident_test --cnfg configs/mini-5,5/test_conf.json
