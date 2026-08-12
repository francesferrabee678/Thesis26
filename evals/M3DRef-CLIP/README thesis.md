# Visual Grounding replication

For visual grounding replication the following steps have to be followed

1. Go to the M3DRefClip repository and follow the set up instructions, make sure that you are able to replicate Lexicon3D's m3drefclip reference numbers. Note that use_gt_proposal=True in the config.

2. run Thesis26/lexicon3d/jobs/matched_script.py, this was created for efficiency. It takes features and premappes them onto the point coordinates used by the M3DRef-CLIP .pth scene files. Since Lexicon3D uses the features directly instead of through an hdf5 file I created lexicon3d_loader which loads the features; matched_script was created to speed this process up. 

3. run Thesis26/lexicon3d/jobs/train_grounding_[model_name].sh this is the train script, after its done it will give the numbers for the model you are running

4. 
for the visualizations run either the test script or the score script(which is a modified version of the test script with per scene confidence scores) following the logic used here lexicon3d/jobs/test_grounding.sh then run the visualization script Thesis26/lexicon3d/jobs/grounding_visual_results.sh 

you can change the models, the type of scene 'Multiple','Unique' as well as the scene name


5. Additional experiments: to run ensemble, first run the scores script (if you have not already)found under Thesis26/lexicon3d/evals/M3DRef-CLIP/scores.py this is a modified version of the test script which gives confidence scores for each bounding box in the scene per text description. In order to run this you need to get the final checkpoint from the training script here Thesis26/lexicon3d/evals/M3DRef-CLIP/output/ScanRefer/[model_name]/training

then run Python Thesis26/lexicon3d/evals/M3DRef-CLIP/evaluate.py which will give a table of results for each combination

To run the 2D off version of grounding, go into the config and Thesis26/lexicon3d/evals/M3DRef-CLIP/config/model/m3dref_clip.yaml and change use_2d_features to False



