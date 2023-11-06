# <center>Color Image Enhancement</center>

資工四 408410098 蔡嘉祥

- date due : 6/5
- date handed in : 5/26
<div style="break-after: page; page-break-after: always;"></div>

## <center>Technical description</center>

使用語言 : Python
- third packages :
  - numpy 
  - opencv (cv2)
如何執行: 
- 下命令 : ```python colorenhance.py``` 即可

### 程式結構:
- ```colorenhance.py``` :
  
  主程式，執行後會產生結果在產生的資料夾 ```./result1/``` 裡面
  - ```test_domain_converting()``` : 
    
    我用來測試我的 domain 轉換的結果是否正確
    - 主要是把 RGB 轉成要的 domain，在轉回來然後輸出，看看轉回來的 RGB 是否正確。
    - 轉回來的 RGB 會寫在 ```./testing1/``` 這個資料夾裡面 

- ```./ColorIMGkit/``` :
  裡面放的是我實踐的 color domain 轉換及 enhance 的方法的程式碼。
  - ```cimgcvtr.py``` : RGB, HSI, L\*a\*b\* 的相互轉換。
  - ```RGBimgIO.py``` : 讀寫 image。
    - 由於 opencv 的 imread, imwrite 是 BGR，所以寫這個用來當IO時RGB BGR互換。
  - ```histEQ.py``` : Histogram Eqaulization
  - ```sharpening.py``` : Laplacian Sharpening
  - ```SaturationADJ.py``` : Saturation adjustment

<div style="break-after: page; page-break-after: always;"></div>

## <center>Experimental results & Disccussions</center>

本報告結果附在 ```./result/``` 裡面

### Histogram Eqaulization 
<img src="./forreport/HistEQ/aloe.jpg" width="48%">
<img src="./forreport/HistEQ/house.jpg" width="48%">
<img src="./forreport/HistEQ/church.jpg" width="48%">
<img src="./forreport/HistEQ/kitchen.jpg" width="48%">

- RGB : 
  
  由於 RGB 較不接近 Basis, 所以有時候直接做 histogram equalization 會出現 pesudo color 的情況。例如: house.jpg 跟 kitchen.jpg
- HSI :
  
  相比於 RGB，在 HSI 的 I 上做比較接近原顏色 (H 不動)。

  不果也能看到 Histogram EQ 在太亮的圖片效果不是很好，例如 house.jpg
- L\*a\*b\*:
  
  從視覺效果上來看，在 L\*a\*b\* 的 L\* 上做 Histogram Eqaulzation 轉回 RGB 後，對比的部分比在 HSI 的 I 上做更為明顯。

<div style="break-after: page; page-break-after: always;"></div>

### Sharpening on RGB

將 R, G, B 三個 channels 分別做 Laplacian operation

- Sharpening on RGB & Histogram Equalization on I

  將做過 sharpening 後的 RGB 轉 HSI 並對 I 做 Histogram Equalization 。

<img src="./forreport/sharpening/aloe.jpg" width="48%">
<img src="./forreport/sharpening/house.jpg" width="48%">
<img src="./forreport/sharpening/church.jpg" width="48%">
<img src="./forreport/sharpening/kitchen.jpg" width="48%">

<div style="break-after: page; page-break-after: always;"></div>

### Gamma Correction on Saturation

使用 $S'=(1*S)^{0.5}$

<img src="./forreport/gmma_a1_05.jpg" width="35%">

從圖可以看到，使用 $(\text{A},\gamma)=(1,0.5)$ 這組參數會將 S 較低，也就是較接近黑的部分拉的較飽和；而 S 越大，改善幅度就越小。


<img src="./forreport/Gamma/aloe.jpg" width="48%">
<img src="./forreport/Gamma/house.jpg" width="48%">
<img src="./forreport/Gamma/church.jpg" width="48%">
<img src="./forreport/Gamma/kitchen.jpg" width="48%">

- Histogram Equalization on I & Gamma Correction on S :
  
  將上述的 Histogram Equalization 跟 Gamma Correction 一起使用在 HSI domain
  - I : Histogram Equalization
  - S : Gamma Correction


<div style="break-after: page; page-break-after: always;"></div>

## <center>Reference</center>
- Ch06 投影片 
- http://www.brucelindbloom.com/index.html?Eqn_XYZ_to_Lab.html
- https://cg2010studio.com/2012/10/02/rgb%E8%88%87cielab%E8%89%B2%E5%BD%A9%E7%A9%BA%E9%96%93%E8%BD%89%E6%8F%9B/

