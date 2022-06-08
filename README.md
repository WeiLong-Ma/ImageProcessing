# ImageProcessing
在QT Designer製作介面並使用C++與OpenCV函式庫編寫簡易的影像處理與人臉辨識。
![圖片2](https://user-images.githubusercontent.com/72548453/172563528-7a123a35-b232-4f4a-a36c-07667f87d60f.png)
---
## 一、開發環境與套件
- Qt Designer
- OpenCV >= 4.0.1
- C++ 17 compiler
- Visual studio 2022

### 軟體安裝
- Qt Designer
```
https://download.qt.io/
```
- OpenCV
```
wget https://github.com/opencv/opencv/archive/3.2.0.zip
```
- Visual studio 2022
```
https://visualstudio.microsoft.com/zh-hant/vs/whatsnew/
```
Setup for Windows 10

---
## 二、目前功能
- ROI感興趣區域裁減
- 影像二值化、直方圖等化
- 色彩空間轉換
- 各類濾波器
- 仿射轉換與透視轉換
- 鏡頭捕捉
- 人臉辨識
- 濾鏡功能(影像疊加)
---
## 三、使用方式
### 座標捕捉
單擊滑鼠左鍵後，顯示於左半部影像點擊座標Label

![image](https://user-images.githubusercontent.com/72548453/172568837-0f994e7d-c765-490e-aee9-0027cc808904.png)
### 透視轉換
雙擊滑鼠左鍵儲存四點座標後，輸出結果

![image](https://user-images.githubusercontent.com/72548453/172569194-c0f99be8-2073-4ccd-bc0d-3c2a5373d329.png)
### 鏡頭與濾鏡功能
可選擇使用載入圖片來指定輸入的影像，或者使用鏡頭捕獲影像，並按下"Q"鍵完成截圖來當作輸入影像
輸入影像指定完畢後，按下右方選擇的濾鏡功能即可實現影像疊加

![image](https://user-images.githubusercontent.com/72548453/172570211-cec00e5d-ddab-45df-b601-c5ac2ccb049f.png)
---
## 四、參考資料
-【OpenCV學習筆記】之仿射變換（Affine Transformation）

https://www.twblogs.net/a/5b7b00e42b7177539c24a869

-Creating Animated Snapchat Filters with Mediaipipe | OpenCV | Python

https://youtu.be/QERl1-ckFr8

-Pig's nose (Face instagram filters) - Opencv with Python

https://youtu.be/IJpTe-1cimE

-【OpenCV】學習(一)關於影像疊加以及原理解釋

https://blog.csdn.net/yb536/article/details/40735821
