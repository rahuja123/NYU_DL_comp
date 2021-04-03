# DL competition Spring 2021

[GCP document](https://newclasses.nyu.edu/access/content/attachment/36f46182-ed32-448e-9e54-af9c2f38d6a1/Announcements/088ccc6d-feac-4955-b775-f3e31f5ba631/gcp_1.0.pdf)

[Competition PDF] (https://newclasses.nyu.edu/access/content/attachment/36f46182-ed32-448e-9e54-af9c2f38d6a1/Announcements/f2962e19-8b28-409d-a6ad-450f8a842d02/Deep_Learning_2021_1.0.pdf)

## GCP Details(Follow this line-by-line):
  - ssh <NYU_NetID>@gw.hpc.nyu.edu
  - ssh <NYU_NetID>@greene.hpc.nyu.edu
  - ssh log-4
  - srun --partition=interactive --account dl18 --pty /bin/bash 
- To check the files:
  - df -h



## Competition Details:
  The dataset, of color images of size 96×96, that has the following structure:
  - 512,000 unlabeled images
  - 25,600 labeled training images (32 examples each for 800 classes)
  - 25,600 labeled validation images (32 examples each for 800 classes).
  
## Schedule:
  - 04/11 23:55: the first leaderboard submission deadline
  - 04/23 23:55: labeling request deadline
  - 04/25 23:55: the second leaderboard submission deadline
  - 05/02 23:55: the final leaderboard submission deadline
  - 05/05 9:30-11:20: virtual poster session
  - 05/09 23:55: paper submission deadline
  
### Labeling:
  We need to select 12,800 images from the 512,000 unlabeled images and send the indices of those images. 


