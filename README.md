# Capsule Network via Bottleneck Block and Automated Gradual Pruning
> A pytorch implementation of "A Novel Effective and Efficient Capsule Network via
Bottleneck Residual Block and Automated Gradual
Pruning".
>
>Since the experimental results in the paper are based on the graphics card (Tesla k40) with low computing power, the experiments at that time were very laborious, and there was no good display of the model results. The calculation of the graphics card (GTX 1060 and 3090) used in this reproduction is higher than the former. The results in the paper will fluctuate, but it does not affect the performance of the overall improved model better than the original capsule network. If there is any problem with the algorithm reproduction, you are welcome to give suggestions.
>
>Paper written by Xiankun Zhang, Yue Sun, Yuan Wang,Zixuan Li, Na Li, Jing Su. For more information, please check out the paper here: https://www.sciencedirect.com/science/article/abs/pii/S0045790619302794
>Fashion_MNIST result:
>![image](https://github.com/starsky68/capsnet_blk_prune/blob/master/results/results.jpg)
# Credits
>Primarily referenced this torch implementations:
>
>https://github.com/cedrickchee/capsule-net-pytorch
>
>https://github.com/IntelLabs/distiller
>
>Reference: https://arxiv.org/abs/1710.09829
