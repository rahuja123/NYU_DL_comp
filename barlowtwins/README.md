Barlow Twins: Self-Supervised Learning via Redundancy Reduction
---------------------------------------------------------------
We followed the barlow twins implementation from FAIR and customised it for our dataset. 

The folder structure is as follows:

We got out best model trained from this command:

```
python3 main.py  --epochs 200 --batch-size 512 --learning-rate 0.3 --lambd 0.0051 --projector 8192-8192-8192 --scale-loss 0.024
```

If you are running on GCP Tesla T4, use this command below to replicate the results. 

```
python3 main.py  --epochs 200 --batch-size 128 --learning-rate 0.7 --lambd 0.0051 --projector 8192-8192-8192 --scale-loss 0.024
```

Training time is approximately 3-4 days. 

From the above file we get resnet50_finalnew.pth as the trained model checkpoint. After that we run finetuning supervised training on the command given below. 

### Evaluation: learning linear probe and also finetuning the remaining network.

Train a linear probe on the representations learned by Barlow Twins. Finetune the weights of the resnet using our labeled dataset. 
```
python3 evaluate.py  /dataset $SCRATCH/checkpoints/barlow/resnet50_finalnew.pth --weights finetune  --epochs 120 --lr-backbone 0.002 --lr-classifier 0.5 --weight-decay 0 --checkpoint-dir $SCRATCH/checkpoints/barlow/finetune 
```
After that we get final model in finetune/checkpoint.pth

We have attached 2 sbatch files to replicate on gcp. Please free to edit that according to your account. Those are 'demo_2gpu.sbatch' and 'lincls_2gpu.sbatch'. First you need to run 'demo_2gpu.sbatch' and then 'lincls_2gpu.sbatch'. 

Finally you need to run the file convert.py which will change the state_dictionary to fit into models.resnet50(num_classes=800).

```
python3 convert.py --checkpoint-path $SCRATCH/checkpoints/barlow/finetune/best_checkpoint.pth 
```

The final model will be saved in the current directory with the name of **barlow_nyu_original.pth**
