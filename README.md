# ABC competition

A general model for Binary Classification (ABC)

* Python version : ```3.12```
* Judge System : [```http://140.116.246.240/```](http://140.116.246.240/)

### 最佳的模型
* 目前最好的模型位於 ```merge.py``` 內，大致就是 ```simpleNN```、```LightGBM```、```XGBoost```以及```catBoost```一起使用，最後將這四個得出的機率，再放入另一個 ```MergeNN``` 模型內去計算。
* 目前分數為：```0.872581```。

### 注意事項
* 在 ```merge.py``` 裡面，資料不進行前處理，反而讓最終的模型分數較高，這讓我很不能理解。
* 在 ```catBoost``` 模型處，如果將 ```categoric_features``` 視為一般的 ```numeric_features``` 來處理，會比較有效率，且兩者執行的分數差異不大。