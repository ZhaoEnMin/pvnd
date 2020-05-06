PVND

Enmin Zhao, Kai LI, Junliang Xing

UCAS, CASIA

Our code is based on Open AI RND(https://github.com/openai/random-network-distillation).

### Installation and Usage
The following command should train an RND agent on Montezuma's Revenge
```bash
CUDA_VISIBLE_DEVICES=0 python run_atari.py --gamma_ext 0.999
```
To use more than one gpu/machine, use MPI (e.g. `mpiexec -n 8 python run_atari.py --num_env 128 --gamma_ext 0.999` should use 1024 parallel environments to collect experience on an 8 gpu machine). 

### [Blog post and videos](https://blog.openai.com/reinforcement-learning-with-prediction-based-rewards/)
