# Script to download dataset used in this project.

- This project uses `FaceForensic++` dataset. This data set's credits lies soly with the original creators of the dataset.
- This notebook uses `faceforensics_download_v4.py` download script provided by original owners of the dataset and can be requested through their GitHub Repository for the dataset at `https://github.com/ondyari/FaceForensics`.
- Original datset contains 1000 images in five categories, i.e.,
  -  youtube
  -  Deepfakes
  -  Face2Face
  -  FaceSwap
  -  NeuralTextures
-  For ur project we are only going to use 100 videos per categories.

## **Note** Please run the following command in the terminal

---

## Download Original videos

- For Ununtu/Mac:
```python
python3 scripts/faceforensics_download_v4.py datasets/FaceForensics++ -d original -c c23 -t videos -n 100 --server EU2
```

- For Windows:
```python
python .\scripts\faceforensics_download_v4.py .\datasets\FaceForensics++\ -d original -c c23 -t videos -n 100 --server EU2
```

---

## Download Deepfakes videos

- For Ununtu/Mac:
```python
python3 scripts/faceforensics_download_v4.py datasets/FaceForensics++ -d Deepfakes -c c23 -t videos -n 100 --server EU2
```

- For Windows:
```python
python .\scripts\faceforensics_download_v4.py .\datasets\FaceForensics++\ -d Deepfakes -c c23 -t videos -n 100 --server EU2
```

---

## Download Face2Face videos

- For Ununtu/Mac:
```python
python3 scripts/faceforensics_download_v4.py datasets/FaceForensics++ -d Face2Face -c c23 -t videos -n 100 --server EU2
```

- For Windows:
```python
python .\scripts\faceforensics_download_v4.py .\datasets\FaceForensics++\ -d Face2Face -c c23 -t videos -n 100 --server EU2
```

---

## Download FaceSwap videos

- For Ununtu/Mac:
```python
python3 scripts/faceforensics_download_v4.py datasets/FaceForensics++ -d FaceSwap -c c23 -t videos -n 100 --server EU2
```

- For Windows:
```python
python .\scripts\faceforensics_download_v4.py .\datasets\FaceForensics++\ -d FaceSwap -c c23 -t videos -n 100 --server EU2
```

---

## Download NeuralTextures videos

- For Ununtu/Mac:
```python
python3 scripts/faceforensics_download_v4.py datasets/FaceForensics++ -d NeuralTextures -c c23 -t videos -n 100 --server EU2
```

- For Windows:
```python
python .\scripts\faceforensics_download_v4.py .\datasets\FaceForensics++\ -d NeuralTextures -c c23 -t videos -n 100 --server EU2
```

---

> *Note: while running the scripts you are in the data directory and script `faceforensics_download_v4.py` is in data/scripts direcory.*