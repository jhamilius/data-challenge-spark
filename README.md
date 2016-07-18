# data-challenge-spark

<hr>

Repository for data-challenge files

## How to execute on cluster

To train the model : 

```
spark-submit --master yarn --num-executors 8 --driver-memory 2g --conf spark.ui.port=7770 code/evaluation.py
```

To generate the predictions.txt file :

```
spark-submit --master yarn --num-executors 8 --driver-memory 2g --conf spark.ui.port=7770 code/classify.py
```

To test the predictions : 

```
spark-submit --master yarn --num-executors 8 --driver-memory 2g --conf spark.ui.port=7770 code/evaluate_F.py
```

## List of files

- evaluation.py : perform model training on training data (main file)
- preProcessing.py : clean the data before training
- extract_terms.py : do some features transformation on the dataset
- helpers.py : other functions
- loadFiles.py : load the data

## Predictions

- Predictions are located in the [predictions.txt](https://github.com/jhamilius/data-challenge-spark/blob/master/predictions.txt) file.




