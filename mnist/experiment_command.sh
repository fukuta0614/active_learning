#!/bin/sh
parallel "python train_active_mnist.py qbc_{} -g 0 --query_by_committee --clustering" ::: 1 2 3
parallel "python train_active_mnist.py random_sample_{} -g 2 --random_sample --label_init random" ::: 1 2 3
parallel "python train_active_mnist.py uncertain_qbc_{} -g 3 --clustering --query_by_committee --uncertain_with_dropout" ::: 1 2 3
parallel "python train_active_mnist.py uncertain_{} -g 3 --clustering --uncertain" ::: 1 2 3
parallel "python train_active_mnist.py uncertain_with_dropout_{} -g 4 --clustering --uncertain_with_dropout" ::: 1 2 3
parallel "python train_active_mnist.py uncertain_qbc_without_clustering_{} -g 5 --query_by_committee --uncertain_with_dropout" ::: 1 2 3
parallel "python train_active_mnist.py uncertain_only_{} -g 5 --uncertain" ::: 1 2 3

parallel "python train_active_mnist.py uncertain_qbc20_aug_epoch100_{} -g 1 --clustering --query_by_committee --uncertain_with_dropout --aug_in_inference --committee_size 30" ::: 1 2 3
parallel "python train_active_mnist.py uncertain20_aug_epoch100_{} -g 2 --clustering --uncertain_with_dropout --aug_in_inference --committee_size 30" ::: 1 2 3
parallel "python train_active_mnist.py uncertain_epoch100_{} -g 0 --clustering --uncertain --epoch_interval 100" ::: 1 2 3
parallel "python train_active_mnist.py uncertain_only_epoch100_{} -g 0 --uncertain --epoch_interval 100" ::: 1 2 3


python train_active_mnist.py qbc_epoch100_1 -g 0 --query_by_committee --clustering --aug_in_inference --epoch_interval 100