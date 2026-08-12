# Thesis replication

In order to replicate the findings of this thesis perform the following steps:

1. Set up the conda environment using all the Lexicon3D requirements (see there repository for more details)

2. clone the repositories for Uniscene3D,Concerto, dinov3, and dinov2 from there respective repository pages. For Pointcept clone and older version so that the weights from HuggingFace work, the version I cloned is commit 2b247ed555e96eae11710dfde79a3fc16c8c74b3 (HEAD, tag: v1.5.0).

3. Download the weights for each model from there respective repositories

4. for each model repository there should be a requirements file for the conda environment, run these as well. You may have to create two different Conda environments,  one of the older models and one for the newer models if there is a clash

5. Download ScanNet and ScanRefer, instructions can be found on their repository pages

6. clone Openscene_scripts  and run the 3D and 2D preprocessing scripts

7. Run the fussion scripts each can be found here Thesis26/lexicon3d/lexicon3d/fusion_scannet_[model_name].py using the job which can be found Thesis26/lexicon3d/jobs/fusion[model_name].sh

8. 

- To visualize the representations using PCA you can run Thesis26/lexicon3d/visualize_feature.py changing the model name and the scene numbers inside the script

- the representational CKA script can be found here: Thesis26/lexicon3d/jobs/cka.py

- Segmentic segmentation part of this repository can be found here: /home/fferrabe/Thesis26/lexicon3d/evals/segmentation
There is another ReadMe thesis file to go through it

- Visual grounding can be found here: Thesis26/lexicon3d/evals/M3DRef-CLIP
again there will be another Read Me thesis file as well


**please note that any code changed from the original repositories will contain the reason at the top of the changed code code. In addition i have kept the old unchanged version and it is marked [code_name]_org **

