NAME=$1
ID_DATASET=CIFAR-100 
ID_LOC=datasets/CIFAR100
OOD_LOC=datasets/small_OOD_dataset


python eval_ood.py \
        --epoch 500 \
        --model resnet34 \
        --head mlp \
        --gpu 0 \
        --score knn \
        --K 300 \
        --in_dataset ${ID_DATASET} \
        --id_loc ${ID_LOC} \
        --ood_loc ${OOD_LOC} \
        --name ${NAME}