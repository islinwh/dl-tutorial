# dl-tutorial
Implementation of LeNet, ResNet18, ResNet50, training and testing on cifar10

Using Python3.6 and Pytorch 1.0

The codes prefer running on gpu over cpu when you have installed cuda.

### How to run the code ?

---

- LeNet	 `python LeNet.py`
- ResNet18   `python ResNet18.py`
- ResNet50   `python ResNet50.py`

you can setting the parameters such as num_epoch, batch_size, learning rate, etc

`--batch_size 256`

`--epoch 100`

`--lr 0.01`

`--scheduler_lr "yes"`

more details can be found in code.

### Performance

---

| Model    | Test Acc on Cifar10 |
| :------- | ------------------- |
| LeNet    | 64.8%               |
| ResNet18 | 83.0%               |
| ResNet50 | 87.8%               |

### Improvement

---

- Pre-train on ImageNet and fine-tune on Cifar10
- Choose Adam instead of SGD
- Change parameters of Normalization: mean = [0.485 0.456 0.406], std = [0.229 0.224 0.225] 
- Adopt data argumentation, such as rotating , cropping image, or adding random illumination
- For conv1, replace 7x7 conv with three 3x3 conv 
- Tune learning rate, batch_size, etc