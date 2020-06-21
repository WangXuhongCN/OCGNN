# -*- coding: utf-8 -*-
from pypai.job.pytorch_job_builder import PytorchJobBuilder
from pypai.utils.km_conf_utils import KubemakerRole, ResourceSpec


def main():
    gpu_resource = ResourceSpec(core=2, memory_m=4096, gpu_num=1, gpu_percent=100, disk_m=8192)
    gpu_master_role = KubemakerRole(count=1, resource_spec=gpu_resource, envs=None)
    image = 'reg.docker.alibaba-inc.com/aii/aistudio:pytorch1.4-20200331144604'
    gpu_builder = PytorchJobBuilder(source_root='./',
                                    command='source init.sh && python main.py --dataset cora --module GCN --nu 0.1 --lr 0.001 --n-hidden 32 --n-layers 2 --weight-decay 0.0005 --n-epochs 4000 --early-stop',
                                    image=image,
                                    cluster="alipay_et15",
                                    pool="kubemaker",
                                    master=gpu_master_role)

    gpu_builder.run()


if __name__ == '__main__':
    main()