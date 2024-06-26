# DeTensor Examples
使用 DeTensor 编写的示例程序
## lr-demo.py 线性回归示例
在本示例中，我们使用Scikit-learn库中的Diabetes dataset数据集用作演示，随机选取80%的数据用作训练，剩余的用作测试。

该数据集包含442个样本和10个特征，输出标签为该患者一年之后患疾病的定量指标。数据集的10个特征包括年龄、性别、体重指数、平均血压以及六个血清测量值（例如，血红蛋白、葡萄糖、胰岛素等）。

在场景中我们共设置有5个参与方，其中3个为数据所有者，分别记作$A\ B\ C$，1个为辅助节点，记作$D$，1个计算发起和结果汇总节点，记作$S$。

其中，参与方$A$拥有该数据集的前5个特征$X_0$，参与方$B$拥有该数据集的后5个特征$X_1$，参与方$C$拥有该数据集的标签$Y$。

我们的目标是：在各参与方所拥有的数据互不暴露的前提下，对该数据集进行去中心化、结果可靠、快速高效的线性回归任务训练。


## psi-demo.py 隐私求交示例
本示例中，要将双方的订单数据进行匹配，但都不能把自己的数据泄露给另一方。我们将数据分别放在A,B两个节点，其中A节点和B节点分别持有`receipt.csv`和`invoice.csv`，另外需要辅助节点C帮助A,B两边数据做隐私求交和隐私比较。

先分别在A,B两边对订单数据进行预处理后，使用隐私求交得出双方共有的数据条目，再使用隐私求交和隐私比较编写的匹配算法，让两边数据依次进行一对一，一对多，多对一，多对多的匹配。最后得到的结果是两边相互匹配的订单数据以及匹配的方式。