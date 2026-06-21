# MNIST Embedded Test

This folder contains a small test harness for checking a trained MNIST model through the CyxWiz inference API.

## Prerequisites

- The CyxWiz engine must be running.
- The MNIST model must be deployed in `Deploy > Embedded`.
- The embedded server must be listening on `http://localhost:8080`.

The default test images are already included in `mnist_data/`.

## Run The Test

From this folder, run:

```powershell
python test_inference.py --endpoint http://localhost:8080/v1/predict --num-samples 20
```

To test more samples:

```powershell
python test_inference.py --endpoint http://localhost:8080/v1/predict --num-samples 100
```

## What The Script Does

- Loads MNIST test images from `mnist_data/`
- Sends each image to the embedded `/v1/predict` endpoint
- Prints running accuracy
- Stops with a summary of correct and incorrect predictions

## Notes

- For Embedded deployment, you do not need to pass `--deployment-id`.
- If you later use Server Node deployment, you can still pass a deployment id explicitly:

```powershell
python test_inference.py --endpoint http://localhost:8080/v1/predict --deployment-id embedded --num-samples 20
```

 python test_inference.py --endpoint http://localhost:8080/v1/predict --num-samples 20 --verbose --log-file mnist_results.csv

 python test_inference.py --endpoint http://localhost:8080/v1/predict --num-samples 3 --verbose

  python D:\Dev\CyxWiz_Claude\examples\python\cats_dogs_embedded_inference.py D:\demo\mrcj\datasets\Cats_Dogs\cat_dog\dog.11289.jpg --url http://localhost:8081/v1/predict --health-url http://localhost:8081/health