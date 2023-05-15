import pandas as pd

meta_batch_size = 20

batch_size = 500
['accuracy', 'ELBO', 'Label_KL', 'Semantic_KL', 'Reconst_Loss', 'CE_Loss']

path_to_train = "/home/dfki.uni-bremen.de/csingh/DFKI/PhysWM/trident_primitives/output/meta_lrng/files/folder/learning_to_meta-learn/logs/TRIDENT_primitives_5-way_4-shot_1-queries/exp1/train.csv"
path_to_valid = "/home/dfki.uni-bremen.de/csingh/DFKI/PhysWM/trident_primitives/output/meta_lrng/files/folder/learning_to_meta-learn/logs/TRIDENT_primitives_5-way_4-shot_1-queries/exp1/valid.csv"
path_to_test = "/home/dfki.uni-bremen.de/csingh/DFKI/PhysWM/trident_primitives/output/meta_lrng/files/folder/learning_to_meta-learn/logs/TRIDENT_test_primitives_5-way_4-shot_1-queries/exp1/test.csv"

df_train = pd.read_csv(path_to_train)
df_valid = pd.read_csv(path_to_valid)
df_test = pd.read_csv(path_to_test)

print("Train")
print(df_train)

print("Test")
print(df_test)

print("Valid")
print(df_valid)
