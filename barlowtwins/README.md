Barlow Twins: Self-Supervised Learning via Redundancy Reduction
---------------------------------------------------------------
We followed the barlow twins implementation from FAIR and customised it for our dataset. 


### Barlow Twins  Unsupervised Training

We got our best model trained from this command:

```
python3 main.py  --data /dataset --epochs 200 --batch-size 512 --learning-rate 0.3 --lambd 0.0051 --projector 8192-8192-8192 --scale-loss 0.024 checkpoint-dir $SCRATCH/checkpoints/barlow
```

If you are running on GCP Tesla T4(Greene) use this command below to replicate the results. 

```
python3  main.py --data /dataset --epochs 200 --batch-size 128 --learning-rate 0.7 --lambd 0.0051 --projector 8192-8192-8192 --scale-loss 0.024 --checkpoint-dir $SCRATCH/checkpoints/barlow
```
or can run this sbatch on gcp

```
sbatch demo_2gpu.sbatch
```


Training time is approximately 5+ days. 

From the above file we get resnet50_finalnew.pth as the trained model checkpoint. After that we run active learning using Core Set and later finetuning supervised training. 

### Core Set Evaluation
We run this command below for Core set. The arguments that are required for the file are as follows: dataset path, pretrained unsupervised model checkpoint, #samples desired, PCA compression. 

We have set #samples desired= 12800, PCA=40 

```
pip install sklearn
pip install Pillow
python Core_set.py  /dataset $SCRATCH/checkpoints/barlow/resnet50_finalnew.pth  12800 40
```
SBATCH FILE version

```
pip install sklearn
pip install Pillow
sbatch  active_image_search.sbatch 
```
*Note: You might have to edit the pretrianed path for the unsupervised trained checkpoint. Ex. 'python Core_set.py /dataset /path_to_checkpoint/resnet50_finalnew.pth 12800 40'*

Search Time is approximately 2-4 Hours. This code will create the file 'request_18.csv' in the current directory of the code. 


### Evaluation: learning linear probe and also finetuning the remaining network.


Train a linear probe on the representations learned by Barlow Twins. Finetune the weights of the resnet using our labeled dataset. 
```
python3 evaluate.py  /dataset $SCRATCH/checkpoints/barlow/resnet50_finalnew.pth --weights finetune  --epochs 120 --lr-backbone 0.002 --lr-classifier 0.5 --weight-decay 0 --checkpoint-dir $SCRATCH/checkpoints/barlow/finetune 
```
or run the sbatch file on gcp

```
sbatch lincls_2gpu.sbatch
```

After this you will get final model in checkpoint_path/finetune/checkpoint.pth

**Training on the EXTRA Dataset IMPORTANT**
- To train on the additional labels. Move both label_18.pt and request_18.csv from this directory to the same directory as the dataset. In the exact same folder as train_label_tensor.pt and val_label_tensor.pt

To train supervised on the extra labels dataset, just use the command given below:
```
python3 evaluate_nyu2.py  /dataset $SCRATCH/checkpoints/barlow/resnet50_finalnew.pth --weights finetune  --epochs 120 --lr-backbone 0.002 --lr-classifier 0.5 --weight-decay 0 --checkpoint-dir $SCRATCH/checkpoints/barlow/finetune_nyu2/
```
After this you will get final model in checkpoint_path/finetune_nyu2/checkpoint.pth

### SBATCH files

We have attached 3 sbatch files to replicate on gcp. Please free to edit that according to your account. Those are 'demo_2gpu.sbatch' , 'active_image_search.sbatch' and 'lincls_2gpu.sbatch'. 

```
sbatch demo_2gpu.sbatch
sbatch active_image_search.sbatch
sbatch lincls_2gpu.sbatch
```

### Converting the network state dict

Finally you need to run the file convert.py which will change the state_dictionary to fit into models.resnet50(num_classes=800).

```
python3 convert.py --checkpoint-path $SCRATCH/checkpoints/barlow/finetune/best_checkpoint.pth 
```

The final model will be saved in the current directory with the name of **barlow_nyu_original.pth** . You can use this model to run eval.py and submission.py.


For any queries, please contact at ra3136@nyu.edu. 
