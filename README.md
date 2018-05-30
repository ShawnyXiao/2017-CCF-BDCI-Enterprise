# 2017-CCF-BDCI-Enterprise

这是我的第一个数据挖掘比赛，CCF 大数据与计算智能大赛（BDCI）中的一题：[**企业经营退出风险预测**](http://www.datafountain.cn/#/competitions/271/intro)。最终取得复赛 **A 榜第 3**，**B 榜第 9** (**Top 1.58%**) 的成绩。

这个比赛 12 月中旬就结束了，硬是被我拖到现在才来总结，我这拖延症真的是……现在回忆起这个比赛，比赛时的那种郁闷感依然记忆犹新。我在复赛的第 5 天便达到了分数 6924，但之后一直无法提分，这种烦躁感当时给我带来了挺大的困扰（当然最后还是提升到了分数 6930）。等比赛结束之后，我回过头来看，其实当时我参赛的心态是不端正的，功利心太强，这样带来的问题就是比赛心态的爆炸，自己的眼界会被约束，提分方式的想象力也会被限制。最好的心态应该是抱着学习的心态参赛，只要能够学到一点点新的东西，就会感到惊喜。

另外一个想说的点是，我们团队在复赛 A 榜中排名第 3，但是切换 B 榜之后，便跌到第 9 了，这个现象直接导致我们团队没有进入决赛，因此我会在后文中谈一谈为什么会有这个现象。

我的另两位队友 JinjinLin 和 ZhenxianZheng 也开源了解决方案，详情请见 [JinjinLin 的解决方案](https://github.com/linjinjin123/2017-CCF-BDCI-Enterprise) 和 [ZhenxianZheng 的解决方案](https://github.com/zhengzhenxian/2017-CCF-BDCI-Enterprise)

## 向导

1. [Why?](#why)
2. [代码框架](#代码框架)
3. [特征](#特征)
4. [模型](#模型)
4. [踩过的坑](#踩过的坑)
5. [没尝试的点](#没尝试的点)
6. [未进决赛的原因分析](#未进决赛的原因分析)
6. [嘿！](#嘿)

## Why?

CCF 举办的这次大赛中这么多比赛，为什么唯独选择这个呢？

1. 因为**门槛低**。我在参赛之前对所有的比赛有过大致的了解，其中比赛类型包括：自然语言处理（NLP）、计算机视觉（CV）和传统的数据挖掘比赛等等。作为一个第一次参赛的新人，我的重心不会放在需要一定的门槛的比赛，因此就排除了 NLP 和 CV 的比赛，再挑一个门槛最低的，那么目标就锁定了，于是我便将重心放在了企业经营退出风险预测这个比赛。
2. 因为**有师兄带（提供 baseline，指导尝试方向）**。今年的 CCF 举办的大赛，我们实验室不少人参赛了，其中也包括不少往年拿过奖的师兄，他们有参赛经验。作为一只菜鸟，自然是希望有人能够给予少走弯路的建议。而师兄也建议新手参加这个方式相对简单的比赛作为入门。

为什么我想要说一下这个呢，因为我相信未来有很多的新人会尝试加入数据挖掘的阵营中，他们也会遇到相同的境遇，我希望能够将我当时的一些思考与选择作为他们的参考选项，以便于他们做出他们的最优选择。

## 代码框架

第一次参赛，可以说连 Python 的语法都不熟悉，更何况 pandas 的各种操作。这时候师兄给的 baseline 就显得十分重要了。当中的各种基础操作，例如：文件读取、数据定义、分组聚集等等，对我来说都是新鲜的。其中最为关键的是**传统的数据挖掘比赛中的代码框架**。我们来看一下，这个极为经典的代码框架（非原始 baseline 框架，我做了一些修改）。

```python
# 1. 导入库
import numpy as np
import pandas as pd
...

# 2. 读取数据文件
train = pd.read_csv('../data/input/train.csv')
test = pd.read_csv('../data/input/evaluation_public.csv')
...

# 3. 定义特征构建函数
def get_entbase_feature(df):
	...
def get_alter_feature(df):
	...
...

# 4. 调用函数，构建特征
entbase_feat = get_entbase_feature(entbase)
alter_feat = get_alter_feature(alter)
...

# 5. 拆分数据集的特征与标签
dataset = pd.merge(entbase_feat, alter_feat, on='EID', how='left')
...
trainset = pd.merge(train, dataset, on='EID', how='left')
testset = pd.merge(test, dataset, on='EID', how='left')
train_feature = trainset.drop(['TARGET', 'ENDDATE'], axis=1)
train_label = trainset.TARGET.values
test_feature = testset
test_index = testset.EID.values

# 6. 模型的交叉验证
...
iterations, best_score = xgb_cv(train_feature, train_label, params, config['folds'], config['rounds'])
...

# 7. 模型的训练与预测
...
model, pred = xgb_predict(train_feature, train_label, test_feature, iterations, params)
...

# 8. 结果文件的写出
res = store_result(test_index, pred, 0.18, '1207-xgb-%f(r%d)' % (best_score, iterations))
```

从上面给的样例代码中，我们可以观察到整个代码的框架如下：

1. **导入库**
2. **读取数据文件**
3. **定义特征构建函数**
4. **调用函数，构建特征**
5. **拆分数据集的特征与标签**
6. **模型的交叉验证**
7. **模型的训练与预测**
8. **结果文件的写出**

使用这样一个代码框架，能够十分清晰的知道整个数据挖掘的流程，这一点对于第一次参赛的新人是尤为重要的。另外当我们想要提分时，我们只需要在特定的部分做出相应的修改就能够达到目的。例如：我希望构建新的特征，来提升我的分数，那么这时只需要新增框架中的第 3 和第 4 部分即可。

## 数据预处理

这个数据集中存在着不少的脏数据，这个阶段便是对这些脏数据进行处理，其中包括：

1. 转化或者移除数据中存在的**中文字符**
2. 针对性的**空值**填充
3. 针对性地去除**重复值**
4. **异常值**的处理（这点我没有做）

## 特征

我将特征分为 5 个部分，分别是**基础特征**、**偏离值特征**、**交叉特征**和**想象力特征**。

### 1. 基础特征

基础特征是比赛中最容易想到的特征，其中包括：

1. **保留字段**。数据集中某些关键字段直接保留成特征，例如：```uid```、```ZCZB```、```RGYEAR```、```INUM```、```ENUM``` 等
2. **统计特征**。以某几个字段作为分组字段，然后进行统计操作，统计操作包括：计数、求和、最小值、最大值、最小最大差值、均值、标准差、比例等
3. **特定集合中的统计特征**。先进行过滤，然后以某几个字段作为分组字段，然后进行统计操作。例如：统计近 1、2、5 年内的修改数额的最小值、最大值和均值等

### 2. 偏离值特征

偏离值特征指**单个个体与分组之间的偏离距离**。以下的代码所生成的特征便是这一类特征：

```python
dataset['MPNUM_CLASS'] = dataset['INUM'].apply(lambda x : x if x <= 4 else 5)
dataset['FSTINUM_CLASS'] = dataset['FSTINUM'].apply(lambda x : x if x <= 6 else 7)
dataset.fillna(value={'alt_count': 0, 'rig_count': 0}, inplace=True)
for column in ['MPNUM', 'INUM', 'FINZB', 'FSTINUM', 'TZINUM', 'ENUM', 'ZCZB', 'allnum', 'RGYEAR', 'alt_count', 'rig_count']:
    groupby_list = [['HY'], ['ETYPE'], ['HY', 'ETYPE'], ['HY', 'PROV'], ['ETYPE', 'PROV'], ['MPNUM_CLASS'], ['FSTINUM_CLASS']]
    for groupby in groupby_list:
        if 'MPNUM_CLASS' in groupby and column == 'MPNUM':
            continue
        if 'FSTINUM_CLASS' in groupby and column == 'FSTINUM':
            continue
        groupby_keylist = []
        for key in groupby:
            groupby_keylist.append(dataset[key])
        tmp = dataset[column].groupby(groupby_keylist).agg([sum, min, max, np.mean]).reset_index()
        tmp = pd.merge(dataset, tmp, on=groupby, how='left')
        dataset['ent_' + column.lower() + '-mean_gb_' + '_'.join(groupby).lower()] = dataset[column] - tmp['mean']
        dataset['ent_' + column.lower() + '-min_gb_' + '_'.join(groupby).lower()] = dataset[column] - tmp['min']
        dataset['ent_' + column.lower() + '-max_gb_' + '_'.join(groupby).lower()] = dataset[column] - tmp['max']
        dataset['ent_' + column.lower() + '/sum_gb_' + '_'.join(groupby).lower()] = dataset[column] / tmp['sum']
dataset.drop(['MPNUM_CLASS', 'FSTINUM_CLASS'], axis=1, inplace=True)
```

这段代码的意思是：

1. 首先，根据分组字段对数据集进行分组
2. 然后计算每个个体与分组的均值、最小值、最大值和求和值之间的偏离距离

这类特征对于这个比赛十分有效，是我分数大幅上升的一个原因。

### 3. 交叉特征

交叉特征指**不单单从一个角度去构建特征，而从多个角度构建够特征**，或者说**将特征之间相互作用后生成新的特征**。这类特征包括：

1. **加减乘除特征**。将特征与特征做加减乘除操作，也就是所谓的暴力出奇迹。例如：```MPNUM+INUM```、```FINZB/ZCZB``` 等
2. **独热交叉特征**。将一些特征做独热编码后，然后乘以某个特征。例如：将 ```HY``` 做独热编码后，乘以 ```ZCZB```、```RGYEAR``` 等
3. **多项式交叉特征**。对特征做多项式组合。例如：```MPNUM^2+INUM``` 等（我没有做这类交叉特征）

交叉特征的效果也十分明显，能显著的提升分数，其中独热交叉特征在这个比赛中最为有效。

### 4. 想象力特征

想象力特征这个词是我自己构造的，指的是根据实际的业务场景，思考其中可能存在的一些隐晦的特征。例如：投资表中，就可以构建一个投资网络，然后基于这个网络提取相关的特征。这个思路来自我的师兄 @Kaho，这也是我赛后才了解到的特征构造方式，十分新颖。

## 模型

模型部分包括：**单模型的提分**与**多模型融合**。

首先，谈谈单模型的提分。在这个比赛中，根据师兄的建议，我选择了 XGBoost，使用它的原因在于：

1. 树模型有较强的可解释性，往往简单且高效
2. 树模型对于异常值有较强的鲁棒性
3. 树模型对特征处理的要求比较低，不需要对特征进行归一化与空值填充

其次，是多模型融合。这部分是我的另一位队友做的，因此我没有过多的尝试多模型融合。在这个比赛中，我们团队的融合效果不是太好，加权融合之后分数仅提升 1 至 2 个千。

## 踩过的坑

新人入赛不踩坑是不可能的，比赛中我是踩了无数个坑，其中比较有意思的，比较隐晦的有这么几个：

1. **不要带着刻板印象去筛选特征**，换句话说，你不要觉得其他比赛没用的特征对于这个比赛同样没用。在这个比赛中，ID 特征是一个强特征，我刚开始就带着刻板印象把它删了，导致 3 个千分点的劣势，发现这个问题也耗费了不少时间
2. **在对 dataframe 排序之后一定要 调用 ```reset_index(drop=True)```**，不然之后对这个 dataframe 的各种操作的是误操作。这个坑同样耗费了我不少的精力
3. **不要太早就开始模型调参**，模型调参只能带来极少的提升，在你的分数没有达到一定竞争力的时候，调参带来的收益是极少的，因此在调参这个举动的价值在比赛早期是较低的
4. 复赛开始后，**初赛数据别果断抛弃**，应该试一试效果，辩证式的采纳

## 没尝试的点

1. 没尝试**融合大法**。因为团队中有队员负责融合，所以在比赛中我没有尝试融合大法，这点比较可惜。另外我们团队的融合策略是 blending（加权融合），还可以尝试的策略包括：stacking、bagging 等
2. 没尝试使用**初赛的数据**。这点输在新人没经验，根本没有意识到可以使用初赛的数据

## 未进决赛的原因分析

我们团队在复赛 A 榜中排名第 3，但是切换 B 榜之后，便跌到第 9 了，这个现象直接导致我们团队没有进入决赛，在赛后我进行了认真的分析与思考，并且与他人探讨，大致总结了几点原因：

1. **未使用初赛提供的数据**。由于我们是新人队伍，使用初赛数据这个套路我们完全没有考虑到，这样就使得其他既使用了复赛数据也使用了初赛数据的队伍能够占据较大优势
2. 我们加权融合的依据是 A 榜的线上分数，这样有极大记录过拟合 A 榜，更好的做法应该是综合考虑线下分数与 A 榜线上分数，以避免出现**过拟合**现象
3. 我们队伍都是来自一个实验室，和队之后，队伍内部有比较多的交流，这可能导致我们的**特征相似度比较大**，这样融合之后的效果不会特别好，因此我们融合值提升了 1 至 2 个千分点

## 嘿！

感谢以下朋友，他们向我输送了一些新的观点：

- @/微笑/:)/wx，他提出：我们团队来自一个实验室，特征可能比较相似，导致融合效果不好

如果您有任何的想法，例如：发现某处有 bug、觉得我对某个方法的讲解不正确或者不透彻、有更加有创意的见解，欢迎随时发 issue 或者 pull request 或者直接与我讨论！另外您若能 star 或者 fork 这个项目以激励刚刚踏入数据挖掘的我，我会感激不尽~