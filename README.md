# Simitate: A Hybrid Imitation Learning Benchmark

![Simitate Overview](images/simitate_overview.png)

* A preprint can be found on [arxiv](https://arxiv.org/abs/1905.06002).
* A short overview video is available on [youtube](https://www.youtube.com/watch?v=EHRgX0_G-j4)
* A [pytorch dataset integration](https://github.com/airglow/simitate_dataset_pytorch) is available

## Abstract

We present Simitate --- a hybrid benchmarking suite targeting the evaluation of approaches for imitation learning. A dataset containing 1938 sequences where humans perform daily activities in a realistic environment is presented. The dataset is strongly coupled with an integration into a simulator.
RGB and depth streams with a resolution of 960x40 at 30Hz and accurate ground truth poses for the demonstrator's hand, as well as the object in 6 DOF at 120Hz are provided.
Along with our dataset we provide the 3D model of the used environment, labeled object images and pre-trained models. 
A benchmarking suite that aims at fostering comparability and reproducibility supports the development of imitation learning approaches.
Further, we propose and integrate evaluation metrics on assessing the quality of effect and 
trajectory of the imitation performed in simulation. Simitate is available on our project website: https://agas.uni-koblenz.de/simitate.

