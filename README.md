# MachineLearningCART
数据挖掘之 CART 决策树

----------------------------------------------

# 数据集
假设现有训练数据集如下（前期先不考虑连续变量，后面会添加）：

|  名称  |   体温   |  表面覆盖  |   胎生   |   产蛋   |  能飞  |  水生  |  有腿  |  冬眠  |  类标记  |
| :----: | :------: | :--------: | :------: | :------: | :----: | :----: | :----: | :----: | :------: |
| 人     | 恒温     | 毛发       |  是      |  否      | 否     | 否     | 是     | 否     | 哺乳类   |
| 巨蟒   | 冷血     | 鳞片       |  否      |  是      | 否     | 否     | 否     | 是     | 爬行类   |
| 鲑鱼   | 冷血     | 鳞片       |  否      |  是      | 否     | 是     | 否     | 否     | 鱼类     |
| 鲸     | 恒温     | 毛发       |  是      |  否      | 否     | 是     | 否     | 否     | 哺乳类   |
| 蛙     | 冷血     | 无         |  否      |  是      | 否     | 有时   | 是     | 是     | 两栖类   |
| 巨晰   | 冷血     | 鳞片       |  否      |  是      | 否     | 否     | 是     | 否     | 爬行类   |
| 蝙蝠   | 恒温     | 毛发       |  是      |  否      | 是     | 否     | 是     | 否     | 哺乳类   |
| 猫     | 恒温     | 皮         |  是      |  否      | 否     | 否     | 是     | 否     | 哺乳类   |
| 豹纹鲨 | 冷血     | 鳞片       |  是      |  否      | 否     | 是     | 否     | 否     | 鱼类     |
| 海龟   | 冷血     | 鳞片       |  否      |  是      | 否     | 有时   | 是     | 否     | 爬行类   |
| 豪猪   | 恒温     | 刚毛       |  是      |  否      | 否     | 否     | 是     | 是     | 哺乳类   |
| 鳗     | 冷血     | 鳞片       |  否      |  是      | 否     | 是     | 否     | 否     | 鱼类     |
| 乌鸦   | 恒温     | 毛发       |  否      |  是      | 是     | 否     | 是     | 否     | 鸟类     |
| 蝾螈   | 冷血     | 无         |  否      |  是      | 否     | 有时   | 是     | 是     | 两栖类   |
| 猫头鹰 | 恒温     | 毛发       |  否      |  是      | 是     | 否     | 是     | 否     | 鸟类     |

------------------------------------------------------------

# 决策树形状
## ID3
针对上面的数据集，如果我们采用了 ID3 进行构建决策树，那么决策树的形状将会如下所示：
```text
表面覆盖
    毛发->胎生
        是->哺乳类
        否->鸟类
    鳞片->水生
        否->爬行类
        是->鱼类
        有时->爬行类
    无->两栖类
    皮->哺乳类
    刚毛->哺乳类
```
这是一个不太好的决策树，为什么这么说呢？首先，表面覆盖这一特征属性下包含了过多的分支状态，其次，当分类为 C 时，可能的路径太多。那么 C4.5 的决策如何呢？

## C4.5
```text
体温
    恒温->产蛋
        否->哺乳类
        是->鸟类
    冷血->表面覆盖
        鳞片->水生
            否->爬行类
            是->鱼类
            有时->爬行类
        无->两栖类
```
现在 C4.5 算法构建出来的决策树比 ID3 构建出的决策树要好了很多，只是仍然会有一些多路径选择的问题。

## CART
经过剪枝后生成的决策树如下：
```text
体温
    恒温->胎生
        是->哺乳类
        Negative->鸟类
    Negative->水生
        Negative->表面覆盖
            鳞片->爬行类
            Negative->两栖类
        是->鱼类
```

------------------------------------------------------------

# Ref
-   http://www.dataguru.cn/article-4720-1.html

------------------------------------------------------------

　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　本人CSDN博客 [点击链接](http://blog.csdn.net/lemon_tree12138)