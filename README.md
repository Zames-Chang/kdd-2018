# KDD-2018
## 問題描述
透過給予北京跟倫敦過去一年的資料，要推論接下來的 48 小時的天氣狀況的比賽，主要是要預測空氣中的PM2.5等有害粒子的濃度

## 特徵工程
除了有害粒子的相關濃度以外，大賽還提供了其他觀測站的風速風壓風向等等其他天氣觀測站的資料，但最困難的點是，每個觀測濃度的天氣站，與觀測風壓或是風向的天氣站他們是不同的站點，如下圖，你幾乎無法使用其他站點所量測資料，因為他們所在的地理位置其實不太一樣。

![](https://i.imgur.com/13AEOMC.png)

這個問題我給出的解決方法是用幾何合成的方式，去把不同站點的資料擬和成跟觀測有害物質站點的地理位置相同的資料

![](https://i.imgur.com/wmvp4mP.png)

然後我們使用 ramdom forest 找出 feature importance 高的相關特徵作為以下模型的 input
## model

![](https://miro.medium.com/max/900/1*evpDWm7Gm0q0_QLwB0QHPw.png)

資料來源:https://towardsdatascience.com/day-1-2-attention-seq2seq-models-65df3f49e263
