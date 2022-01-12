# Notice: 
In our project, we use a part of source code from [link](https://github.com/yaoing/dan)
# Preparation <br />
- Download pre-trained model of [MSCeleb](https://drive.google.com/file/d/1H421M8mosIVt8KsEWQ1UuYMkQS8X1prf/view) and move the file to ./models
- Download [RAF-DB dataset](https://drive.google.com/drive/folders/1SLuRWt0IjBO2D7UgLeD06CRGV_unx4LW) and extract the raf-basic dir to ./datasets/raf-basic
- Download [Fer2013 dataset](https://www.kaggle.com/msambare/fer2013?select=train) and extract to ./datasets/fer2013
- Run python ./utils/dataprocess_fer.py to process fer2013 dataset <br />

# Training <br/>
- Training with model DAN and fer2013 dataset<br/>
```python
python dan_fer.py --batch_size 64 --lr 0.01
```
- Training with model VGG16 and fer2013 dataset<br/>
```python
python vgg_fer.py --batch_size 32 --lr 0.001
```
- Training with model DAN and rafdb dataset<br/>
```python
python dan_rafdb.py --batch_size 32 --lr 0.01
```
- Training with model DAN and fer2013 dataset<br/>
```python
python vgg_rafdb.py --batch_size 32 --lr 0.001
```
# Pretrain<br />
Pre-trained models can be download for evaluation as following:<br />
| task | epochs | accuracy | link |
| :--: | :-: | :-: | :--: |
| dan_fer | 11 | 69.32 | [link](https://drive.google.com/file/d/1WX739EJWnEylqGAMeeDwJMtzEQpR7QO3/view?usp=sharing) |
| dan_rafdb | 21 | 89.70 | [link](https://drive.google.com/file/d/18XX46_T5pAwOY4zyrJC4QRl1u9HuuLaA/view?usp=sharing) |

# Testing<br >
Run code to test:<br>
```python
python demo.py --image test_image_path
```
