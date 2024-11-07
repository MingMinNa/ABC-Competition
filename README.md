# ABC competition

A general model for Binary Classification (ABC)  

成功大學課程 **[資訊科學概論]** 上的微競賽。  

##### 競賽說明
* 給定 49 個二元分類的部分資料集 
* 希望讓我們能夠寫出一份程式碼，使得其可以在這 49 個資料集中，都能夠產生效果不錯的模型（泛化性高）。

##### 資料集說明
* 資料集內有 ```數值型資料``` 和 ```類別型資料```。當然地，每個資料集的特徵數量都不一樣，有些甚至沒有類別型資料
* ```X_train.csv``` 為訓練資料集的特徵。
* ```y_train.csv``` 為訓練資料集的標籤。
* ```X_test.csv``` 為測試資料集的特徵。
* ```y_predict.csv``` 為測試資料集的結果。其結果為「該筆資料為 1 的機率」，而並非是與 ```y_train.csv``` 一樣的標籤。

##### 競賽評分
* 本次競賽評分方式採用 ```AUC```做為指標。簡單來說，就是 ```AUC``` 越高，則模型區分能力越好。
* 透過將預測結果上傳到 [評分網站](http://140.116.246.240/)，評分網站會回應一個分數。
   > 競賽期間：只會比較資料集中 ```public data```，從而得出分數並顯示。  
   > 競賽結束：會比較全部資料，從而得出分數並顯示。

   > **[Public vs. Private leaderboards]**  
   > Public leaderboard is based on random 1/2 test data you are submitting while the private leaderboard is based on the remaining 1/2 test data. What you see during the competition is the public leaderboard.  
   > The private leaderboard remains secret until the end of the competition. It will determine the final competition ranking!

其餘細節請看 ```DS_CP1_intro.pdf```

### 注意事項
* Python version : ```3.12```