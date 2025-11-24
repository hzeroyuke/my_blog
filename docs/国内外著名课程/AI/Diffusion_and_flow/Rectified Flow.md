[Rectified Flow：矫正流生成式模型的概念及应用实践｜青稞Talk 31期_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1pqHezrED5/?spm_id_from=333.1387.favlist.content.click) 

上面这个视频是Rectified Flow的作者之一的一个talk，非常清晰地讲述了Flow Matching和Rectified Flow的概念，绕过了Diffusion，ODE/SDE等一系列复杂的数学概念，直观的切入了Flow的概念

![](Pasted%20image%2020251002003049.png)

上图是他在引入部分，对于整个生成模型领域的一个简单的概括
早期

- Normalizing Flow
- GAN
- VAE
- EBM

现阶段

- AutoRegressive
- Diffusion

## 1. ReFlow 理论

接下来我们来看Diffusion为代表的一系列生成模型，其核心思想就是将一个分布（一般是高斯）转换成另一个分布（真实数据代表的分布，比如图片，蛋白质分子等等）

![image](Pasted%20image%2020251002003320.png)

最简单的情况如上，比如真实分布只有一个点，那么整个过程很好学习，一切都移动向同一个点即可，下面整个式子就是我们的Loss

但是当我们的分布复杂一点，比如有两个点的时候，就会出现一点问题，下图中我们会希望初始分布尽可能移动向这两个点

![image](Pasted%20image%2020251002003538.png)

这个时候解决这个Loss，我们的分布会变成什么样，实际会变成下图这样，因为我们此时的移动是一条直线，其会变成两个点的平均，这也是我们在VAE中经常看到的结果

![image](Pasted%20image%2020251002003717.png)      

上述问题出在哪呢，原因在于我们强制了这个移动是一条直线，也即其速度的方向是不变的，如果我们去掉这个前提，再来看看（其实这个过程就是把离散转换成连续的过程，我们之前假设的情况并非移动，而是跃迁，移动本身应当是连续的）

因为我们在计算机上不能将运动过程无限微分，我们会选择进行高斯插值来模拟连续的运动

![image](Pasted%20image%2020251002004015.png)

那么我们求解上述结果之后，我们的速度场平均下来是这样的

![image](Pasted%20image%2020251002004135.png)

这个结果就非常符合我们的目标，这也是说明为什么我们在做Diffusion或是Flow的时候要分多步，而不是一步到位的原因，一步到位的移动往往会导致我们采样到真实数据分布的一个平均

我们来看一个更加明显地ODE地例子

![image](Pasted%20image%2020251002004341.png)

在上图中，我们对比了真实地ODE和学习出来地Flow的结果，很明显可以观察到交叉点的不同，学习到的Flow中，左下角的紫色分布有时会向中间移动一段距离，再往右下角的红色分布进行移动

站在上帝视角我们可以很直观的发现，左下角的紫色分布，他就适合移动到右下角，左上角的紫色分布，就适合移动到右上角，但是再训练中这一点是未知的，我们只能尽可能地让紫色分布的全部都往红色分布移动，这也导致了训练出来的结果虽然好，但是走了弯路。理想的情况应该是下图第三张

![image](Pasted%20image%2020251002005120.png)

这时候存在一个解决方案，那就是再训练一段。Rectified Flow相比于Flow Matching最大的区别就是在常规的Flow matching结束后，多了一段训练过程，该训练过程中使用的数据是一阶段训练好的Flow Model采样出来的

经过第二阶段的训练之后的Flow Model，可以做到快速采样，因为这个过程完成了去交叉点的内容

![image](Pasted%20image%2020251002004835.png)

这张图直观的给我们展示了Recified Flow的威力，未经历过二阶段训练的模型，一开始会给我们去噪出一个图像的平均值，这个平均值往往就是一个交叉点，需要多步才能实现准确的生成，而经过Rectified的模型，从初始状态就知道自己该往哪个方向走，因此迅速就能够生成想要的方向，当然多步也会做的更好


## 2. ReFlow 代码

![](asset/Pasted%20image%2020251118204022.png)

这是我在Minst数据集上的训练结果，因为数据集比较简单，所以在2步以上的去噪中就显得区别不大，但是很明显的，ReFlow的模型可以在第一步去噪中就得到很好的结果

具体的代码如下，基本的训练和flow model没有什么区别，仅仅是数据上使用了flow model生成的数据

```python
@torch.no_grad()
def generate_paired_data(model, num_pairs, device="cuda"):
    """
    使用预训练的flow matching模型生成配对数据 (x0, x1, label)
    x0: 初始噪声
    x1: 生成的图像（对应真实数据分布）

    Rectified Flow的核心思想：
    - 使用预训练模型从噪声生成样本，得到轨迹的起点和终点
    - 重新训练模型学习这些起点和终点之间更直的路径
    """
    model.eval()

    x0_list = []
    x1_list = []
    label_list = []

    # 分批生成，避免内存溢出
    num_batches = (num_pairs + batch_size - 1) // batch_size

    print(f"开始生成 {num_pairs} 对配对数据...")

    for i in range(num_batches):
        current_batch_size = min(batch_size, num_pairs - i * batch_size)

        # 随机采样噪声作为起点 x0
        x0 = torch.randn(current_batch_size, channels, image_size, image_size, device=device)

        # 随机采样标签
        labels = torch.randint(0, num_classes, (current_batch_size,), device=device)

        # 使用ODE求解器从x0生成x1
        def ode_func(t: torch.Tensor, x: torch.Tensor):
            t_expanded = t.expand(x.size(0))
            vt = model(x, t_expanded, labels)
            return vt

        t_eval = torch.tensor([0.0, 1.0], device=device)
        trajectory = odeint(ode_func, x0, t_eval, rtol=1e-5, atol=1e-5, method='dopri5')
        x1 = trajectory[-1]  # 取最终时刻的状态

        x0_list.append(x0.cpu())
        x1_list.append(x1.cpu())
        label_list.append(labels.cpu())

        if (i + 1) % 10 == 0:
            print(f"已生成 {(i + 1) * batch_size} / {num_pairs} 对数据")

    # 拼接所有批次
    x0_all = torch.cat(x0_list, dim=0)[:num_pairs]
    x1_all = torch.cat(x1_list, dim=0)[:num_pairs]
    labels_all = torch.cat(label_list, dim=0)[:num_pairs]

    print(f"配对数据生成完成！形状: x0={x0_all.shape}, x1={x1_all.shape}, labels={labels_all.shape}")

    return x0_all, x1_all, labels_all
```