# Power-Aware Training for Printed Neuromorphic Circuits

This github repository is for the paper at ICCAD'23 - Power-Aware Training for Printed Neuromorphic Circuits

cite as
```
Power-Aware Training for Printed Neuromorphic Circuits
Zhao, H.; Pal, P.; Hefenbrock, M.; Beigl, M.; Tahoori, M.
2023 International Conference on Computer-Aided Design (ICCAD), October, 2023 IEEE/ACM.
```



Usage of the code:

1. Training of printed neural networks

~~~
$ sh experiment_power.sh
~~~

Alternatively, the experiments can be conducted by running command lines in `experiment_power.sh` separately, e.g.,

~~~
$ python3 experiment.py --DATASET 0  --powerestimator power  --powerbalance 0.0   --projectname Power-Aware-Training
$ python3 experiment.py --DATASET 0  --powerestimator power  --powerbalance 0.02  --projectname Power-Aware-Training
...
~~~



2.   After training printed neural networks, the trained networks are in `./Power-Aware-Training/model/`, the log files for training can be found in `./Power-Aware-Training/log/`. If there is still files in `./Power-Aware-Training/temp/`, you should run the corresponding command line to train the networks further. Note that, each training is limited to 48 hours, you can change this time limitation in `configuration.py`



3.   Evaluation can be done by running the `evaluation_power.sh` in `./Power-Aware-Training/` folder with

~~~
$ sh evaluation_ICCAD_2022.sh
~~~

 Of course, each line in this file can be run separately as in step 1.
