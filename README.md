# UPF-Hand-Written-Text-Recognition

On this repository you have all the files that you need to run the code.

The dataset is available in the following link: http://www.fki.inf.unibe.ch/databases/iam-handwriting-database

you have to download the zip and store the files in  a folder called "data". The path should be: data/words/ (all the folders and the file "words.txt")

The following files should be in a folder called src, in the same level as "data":
- analyze.py
- DataLoader.py
- PreprocessImage.py	
- TFWordBeamSearch.so
- demo.py (script for showing images in the demo)
- demo1.py (script for infering the images to text)
- trainingScript.py (script for training)

And in a folder called "Model", you should have this files
- wordCharList.txt	
- accuracy.txt
- snapshot-8.index
- snapshot-8.meta	
- checkpoint

If you have any problem you can contact us, or see the example in the repository:
https://github.com/githubharald/SimpleHTR

Thanks,
Joan Rodriguez
Mar Ferrer
