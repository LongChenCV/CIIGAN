### Controllable Illumination Invariant GAN for Diverse Temporally-Consistent Surgical Video Synthesis

This is the official implementation of our CIIGAN for unpaired synthesis of view-consistent surgical video sequences.

<p align="center">
  <strong style="letter-spacing: 500px;">Input</strong>&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp
  <strong style="letter-spacing: 100px;">Baseline</strong>&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp
  <strong style="letter-spacing: 100px;">Ours</strong>
</p>
<p align="center">
  <img src="https://github.com/LongChenCV/CIIGAN/blob/main/syn1.gif" alt="AAA" width="250"/>
  <img src="https://github.com/LongChenCV/CIIGAN/blob/main/baseline1.gif" alt="Figure 2" width="250"/>
  <img src="https://github.com/LongChenCV/CIIGAN/blob/main/ours1.gif" alt="Figure 3" width="250"/>
</p>




<p align="center">
  <strong style="letter-spacing: 500px;">Input</strong>&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp
  <strong style="letter-spacing: 100px;">Ours</strong>
</p>
<div align=center>
<img src="https://github.com/LongChenCV/CIIGAN/blob/main/vid1.gif" width="300" height="200"/>   <img src="https://github.com/LongChenCV/CIIGAN/blob/main/vid_syn1.gif" width="300" height="200"/>
    
<img src="https://github.com/LongChenCV/CIIGAN/blob/main/vid2.gif" width="300" height="200"/>   <img src="https://github.com/LongChenCV/CIIGAN/blob/main/vid_syn2.gif" width="300" height="200"/>

<img src="https://github.com/LongChenCV/CIIGAN/blob/main/vid3.gif" width="300" height="200"/>   <img src="https://github.com/LongChenCV/CIIGAN/blob/main/vid_syn3.gif" width="300" height="200"/>


</div>

We provided the source code, the model trained on public dataset ChoSeg8K, and 10 3D simulation scenes constructed by us using Blender.

### Conda Environment Setting
```
conda create --name CIIGAN python=3.7
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
```

### Blender Environment Setting
Download the blender-2.83.20-linux-x64.tar.xz
```
sudo tar -xf blender-2.83.20-linux-x64.tar.xz
```
Note: Do not use the original python bundled with blender-2.83.20: blender-2.83.20-linux-x64/2.83/python;
Use the python in the CIIGAN conda env by adding the codes into python script:
```
import sys 
sys.path.append('/anaconda3/envs/CIIGAN/lib/python3.7/site-packages/')
```

### Run Quick Demo
We release the model trained on the public dataset 3D-SimUCL+ChoSeg8K, the model and example test data can be downloaded from
https://1drv.ms/f/c/bcdaf3fbecba991b/EtgZX8TRroBDrF_YsVw9-FEB-cvVfI4xKAPsUafA1xS4tg?e=fYzkcD

Run the following commands to quick test our CIIGAN:
```
cd translation_model
python translate.py 
```
The results will be saved in the directory test_output

### Generating Training and Testing Data
The 3D scenes locate in CIIGAN/simulated_data_generation/ and can be visualized in Blender.
ExampleScene_FakeLiver.blend is the public 3D scene.
Scene1, Scene2, …, Scene10 are our constructed 3D scenes. 

Image, Texture and Mask Generation
```
blender-2.83.20-linux-x64/blender Scene1.blend -b -P renderRandomImages.py -- --images 100 --test_render --texture_patch_size 512
```
This data will be saved to ```data/simulated_images/```. 

Video Sequence, Texture and Mask Generation
```
blender-2.83.20-linux-x64/blender Scene1.blend -b -P renderSequences.py -- --test_render --texture_patch_size 512
```
This data will be saved to ```data/simulated_sequences/```. 

CII Images Generation
```
python CIIGAN/translation_model/RGB2CII.py
```

### Training

After data generation, run the following to train the model:
```
cd translation_model
python train.py --output_path data/Surgical/output/
```

### Testing

After training, run the following to test the model:
```
cd translation_model
python translate.py 
```

### Citation

This paper is under review now. If you use this code, please cite our paper later:

```
@InProceedings{CIIGAN_2024,
    author    = {Long Chen, Mobarakol Islam, Matt Clarkson and Thomas Dowrick},
    title     = {Controllable Illumination Invariant GAN for Diverse Temporally-Consistent Surgical Video Synthesis},
}
```

### License

Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
Copyright (C) 2024 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
These licenses allow you to use, modify and share the project for non-commercial use as long as you adhere to the conditions of the license above.

### Contact

If you have any questions, do not hesitate to contact us: ```chenlongcv@gmail.com```
