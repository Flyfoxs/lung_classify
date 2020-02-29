

# Input
ln -s /share/data/lung/lung_xray input


fastai

# 激活函数默认根据损失函数来确定

    fastai 在预测时, 动态根据损失函数来决定激活函数, 相关代码如下: activ = ifnone(activ, _loss_func2activ(self.loss_func))
    
    模型计算loss的时候, 后端没有使用softmax函数, softmax函数是嵌入到了CrossEntropyLoss