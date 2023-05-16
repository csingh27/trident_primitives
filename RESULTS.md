Train-test-validation split: 60% (150), 20% (50), 20% (50)

path_to_train = "/home/dfki.uni-bremen.de/csingh/DFKI/PhysWM/trident_primitives/output/meta_lrng/files/folder/learning_to_meta-learn/logs/TRIDENT_primitives_5-way_4-shot_1-queries/exp1/train.csv"

path_to_valid = "/home/dfki.uni-bremen.de/csingh/DFKI/PhysWM/trident_primitives/output/meta_lrng/files/folder/learning_to_meta-learn/logs/TRIDENT_primitives_5-way_4-shot_1-queries/exp1/valid.csv"

path_to_test = "/home/dfki.uni-bremen.de/csingh/DFKI/PhysWM/trident_primitives/output/meta_lrng/files/folder/learning_to_meta-learn/logs/TRIDENT_test_primitives_5-way_4-shot_1-queries/exp1/test.csv"

Training steps - 150 (NaN errors appear for larger number of steps)

Predictions:  tensor([[0.1307, 0.1406, 0.1752, 0.4298, 0.1236],
        [0.1414, 0.2112, 0.1315, 0.2654, 0.2505],
        [0.1377, 0.2602, 0.1490, 0.1358, 0.3172],
        [0.4787, 0.1296, 0.1320, 0.0769, 0.1829],
        [0.1650, 0.2361, 0.1472, 0.2667, 0.1849]], device='cuda:0',
       grad_fn=<SoftmaxBackward0>)
Targets:  tensor([3, 2, 4, 0, 1], device='cuda:0')

Accuracy = (sum(predictions == targets)) / (targets.size(0))

Accuracy = (Number of correct predictions) / (Total number of examples)

Iterations = 150
batch size = 1

Training step in CSV = each batch

Case 1 - 
n_ways: 5
k_shots: 5
q_shots: 1
Accuracy: 0.46357617019028 (Average over 150)

Case 2 - 
n_ways: 5
k_shots: 1
q_shots: 1
Accuracy:  (Average over 150)

Case 3- 
n_ways: 20
k_shots: 5
q_shots: 1
Accuracy: (Average over 150)

Case 4- 
n_ways: 20
k_shots: 1
q_shots: 1
Accuracy: (Average over 150)