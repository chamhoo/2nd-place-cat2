# 2nd-place-solution-Cat2 [xDeepFM Yes!]

感谢 Categorical Feature Encoding Challenge II 竞赛的组织者, 感谢竞赛参与者之间的慷慨分享和讨论, 这对我帮助很大.  
我将尝试描述一下我的整个竞赛过程:

### 开始竞赛 & 早期的尝试  
 当我最开始参加此次竞赛的时候, 我看到很多参与者说简单的LogisticRegression的效果很好, 所以我使用了sklearn.linear_model.LogisticRegression模型,使用hyperopt进行超参数搜索，之后40-folds训练模型作为baseline.之后尝试了catboost,但是效果并不好. 我发现在数据集中, 离散的分类特征占比很大, 因为我过去有使用Wide&deep 和 DeepFM 的经验，所以我觉得使用一些针对CTR的深度学习模型也许有更好的效果, 事实证明这是一个正确的选择.

### 尝试CTR
Part of the model I use comes from the * deepctr * package, source:
- https://github.com/shenweichen/DeepCTR  
你可以使用 **pip install deepctr** 安装.


**FE:** 我并没有过多将注意力集中在feature engineering上，所以只是对数据集进行了一些主流的改变.  

```python
data_df['num_null'] = data_df.isna().sum(axis=1)
data_df['ord_5_0'] = data_df['ord_5'].str[0]
data_df['ord_5_1'] = data_df['ord_5'].str[1]
```

在这个阶段, 我首先用 TensorFlow 重写了LogisticRegression(batch_size=8192, optimizer='Adam', regularizers=l2)作为新的baseline.

之后,我尝试了多种 ctr 模型, 如果没有特殊说明, 那么都是使用hyperopt在训练集上5折交叉验证进行超参数搜索, 选择出5个表现最好的超参数组合, 最后对每个超参数组合使用40折交叉验证之后取平均数得出score:  

**FM:** pure FM without linear, score: 0.78661  
**DeepFM:** 10个超参数组合, 20折交叉验证，score: 0.78922. 感谢@Siavash精彩的kernel -https://www.kaggle.com/siavrez/deepfm-model, 你在训练deepfm中使用Mish和CyclicLR的思路对我的影响很大.  
**CIN(XdeepFM):** 未使用deep和linear模块, score: 0.78940  
**CrossNet(Deep&Cross):** 未使用deep和linear模块, score: 0.78888  
**afm:** score: 0.78817  
**autoint:** score: 0.78926  

我之所以在CIN和Cross时不使用deep和linear模块，一是因为我想看到这两个模块单独的表现,二是因为我希望之后对表现好的模型进行stacking, 而stacking更偏爱优秀且不同的模型.

可以看出CIN在训练集上的表现最好，之后我将这些模型都提交进Leaderboard之后，验证了我这个判断.但有趣的是, 当我对所有模型作了stacking之后，虽然训练集上分数升高, 但是Leaderboard上分数却显著下降了.考虑到有可能的过拟合风险,我放弃了stacking的想法.  

### Focus on XdeepFM
我开始着重训练xdeepfm.通过和之期一样的方法.得到了一个更好的分数：0.78955, 但问题也随之而来了, 可以看出xDeepFM在训练集上的表现最好，但当我把这次训练出来的结果提交到LB之后，分数不上升反到下降了。

所以我该相信什么?训练集上交叉验证得出的分数还是Public Leaderboard?   

我仔细分析了一下竞赛规则, 训练集拥有600k行数据, 测试集拥有400k行数据, 而 Public Leaderboard 只使用了 25% of the test data,既100k行。当我使用5折交叉验证的时候,每折的validation dataset为120k行, 比较接近Public Leaderboard所使用的数据量, 并且同一个模型的不同折之间validation score经常在[0.790 到 0.787]之间跳转, 方差比较大。最关键的是.当使用相同的kFold迭代器（固定KFold种子和折数）时,不同模型的validation score拥有相关性, 这也就意味着数据集中有一部分样本就是在任何模型上难以训练，甚至无法训练的. 根据此我猜测, Public Leaderboard之所以分数比k折交叉验证的分数低那么多, 很可能只是PLB中这部分样本的占比稍微有点多而以。而整个test data行数与train data 差不多,所以最终结果应该更接近k折交叉验证的分数.

最后，我stacking了CIN和xdeepfm, 在k折交叉验证和PLB上分数都有提高. 在PLB上排名79名.但是那时我相信最终结果会更加接近train set的分数,事实证明这个猜测应该是对的.
