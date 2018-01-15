#!/bin/sh
parallel "python train_active_camelyon.py qbc_{} -g 0 --query_by_committee --clustering" ::: 1 2 3
parallel "python train_active_mnist.py random_sample_{} -g 2 --random_sample --label_init random" ::: 1 2 3
parallel "python train_active_mnist.py uncertain_qbc_{} -g 3 --clustering --query_by_committee --uncertain_with_dropout" ::: 1 2 3
parallel "python train_active_mnist.py uncertain_{} -g 3 --clustering --uncertain" ::: 1 2 3
parallel "python train_active_mnist.py uncertain_with_dropout_{} -g 4 --clustering --uncertain_with_dropout" ::: 1 2 3
parallel "python train_active_mnist.py uncertain_qbc_without_clustering_{} -g 5 --query_by_committee --uncertain_with_dropout" ::: 1 2 3
parallel "python train_active_mnist.py uncertain_only_{} -g 5 --uncertain" ::: 1 2 3

parallel "python train_active_mnist.py uncertain_qbc_aug_{} -g 1 --clustering --query_by_committee --uncertain_with_dropout --aug_in_inference" ::: 1 2 3


python train_camelyon_fromtif.py cbp -g 2 -b 128 --cbp
python train_camelyon_fromtif.py googlenet -g 2 -b 128 --no-texture

python train_active_camelyon.py random_sample_1 -g 0 --random_sample --label_init random --epoch_interval 50
python train_active_camelyon.py random_sample_2 -g 1 --random_sample --label_init random --epoch_interval 50

python train_active_camelyon.py uncertain_1 -g 2 --uncertain --clustering --committee_size 1 --epoch_interval 50
python train_active_camelyon.py uncertain_2 -g 3 --uncertain --clustering --committee_size 1 --epoch_interval 50
python train_active_camelyon.py uncertain_3 -g 4 --uncertain --clustering --committee_size 1 --epoch_interval 50

python train_active_camelyon.py uncertain_qbc_aug_1 -g 5 --clustering --query_by_committee --uncertain_with_dropout --aug_in_inference --epoch_interval 50
python train_active_camelyon.py uncertain_qbc_aug_2 -g 6 --clustering --query_by_committee --uncertain_with_dropout --aug_in_inference --epoch_interval 50
python train_active_camelyon.py uncertain_qbc_aug_3 -g 7 --clustering --query_by_committee --uncertain_with_dropout --aug_in_inference --epoch_interval 50
