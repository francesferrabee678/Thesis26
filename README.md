# Readme
## Abstract of Thesis
A wide range of encoders have been developed for 3D scene understanding, but it remains unclear how their representations transfer to different tasks. Lexicon3D addresses this gap by introducing a unified probing framework to compare 3D encoders on a variety of different 3D scene understanding tasks. However, Lexicon3D only evaluates a small portion of the encoders available today, leaving newer encoder architectures and encoder families unexplored. This thesis extends the Lexicon3D framework by comparing four additional encoders on 3D semantic segmentation and 3D visual grounding: DINOv3(2D to 3D lifted), Point Transformer v3 with Point Prompt Training (3D native), Concerto (3D Self-supervised) and Uniscene3D(CLIP-based). These models were evaluated on ScanNet for 3D semantic segmentation and ScanRefer for 3D visual grounding using the same metrics as Lexicon3D. This thesis also analyzes the learned representations by using PCA visualizations in addition to linear pairwise CKA. The results show that there is no single encoder that is best on both tasks with the relative performance of the encoders changing depending on the task. The best overall encoder is PTv3+PPT placing second on both tasks. In contrast, Uniscene3D and Concerto each excel at one task (Uniscene3D in 3D visual grounding and Concerto in 3D semantic segmentation) but perform poorly on the other. The representational analysis further indicates that models that perform well on semantic segmentation tend to learn more consistent features while models that perform well on visual grounding are more likely to preserve more fine-grained details. Overall this thesis demonstrates the importance of considering both the representation learned by the encoder and the requirements of the specific task when selecting a 3D encoder.

Link to full thesis with references: [Probing diverse 3D encoder representations for scene understanding](https://github.com/francesferrabee678/Thesis26/blob/main/Masters%20Thesis.pdf)

## To replicate the thesis

In order to replicate the findings of this thesis perform the following steps:

1. Set up the conda environment using all the Lexicon3D [requirements](https://github.com/francesferrabee678/Thesis26/blob/main/requirements.txt)

2. clone the repositories for [Uniscene3D](https://github.com/Yebulabula/UniScene3D),[Concerto](https://github.com/Pointcept/Concerto), [Dinov3](https://github.com/facebookresearch/dinov3), and [Dinov2](https://github.com/facebookresearch/dinov2) from their respective repository pages. For Pointcept the older version has been cloned and added to this repository so that the weights from HuggingFace work. The version I cloned is commit 2b247ed555e96eae11710dfde79a3fc16c8c74b3 (HEAD, tag: v1.5.0).

3. Download the weights for each model from their respective repositories

4. For each model's repository there should be a requirements file for the conda environment, run these as well. You may have to create two different Conda environments,  one of the older models and one for the newer models if there is a clash

5. Download ScanNet and ScanRefer, instructions can be found on their repository pages: [ScanNet](https://github.com/scannet/scannet), [ScanRefer](https://github.com/daveredrum/ScanRefer)

6. Clone [Openscene_scripts](https://github.com/pengsongyou/openscene/blob/main/scripts/preprocess/README.md) and run the 3D and 2D preprocessing scripts

7. Run the fussion scripts for each model that can be found here Thesis26/lexicon3d/lexicon3d/fusion_scannet_[model_name].py using the corresponding job Thesis26/lexicon3d/jobs/fusion[model_name].sh

8. 
  - To visualize the representations using PCA you can run Thesis26/lexicon3d/visualize_feature.py changing the model name and the scene numbers inside the script
  
  - The representational CKA script can be found here: Thesis26/lexicon3d/jobs/cka.py
  
  - Segmentic segmentation part of this repository can be found here: /home/fferrabe/Thesis26/lexicon3d/evals/segmentation
and the Visual grounding part can be found here https://github.com/francesferrabee678/Thesis26/blob/main/evals/M3DRef-CLIP/. Read the "ReadMe thesis" files to understand what to do next.

## Acknowledgements
This repo is primarily built based on the work of [Lexicon3D](https://github.com/yunzeman/lexicon3d) as well as [M3DRefCLIP](https://github.com/3dlg-hcvc/multi3drefer). 


