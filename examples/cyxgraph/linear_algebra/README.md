# Sensor calibration (linear algebra nodes)

**Graph:** `sensor_calibration.cyxgraph`. **Docs:** [Linear algebra](../../../docs/usage/linear_algebra.md).

Three sensors (`sensors.csv`: `s1`, `s2`, `s3`) and a reference meter
(`meter.csv`: `meter`) are read 200 times. The data is made up for this
example, so no download is needed. The meter reads 2.0 s1 − 0.5 s2 + 1.5 s3
plus a little noise. The graph finds those weights back, and maps the sensors
onto their two widest directions:

```
Sensors -+-> Calibration weights (Linear Solve, least squares) <- Meter (its b input)
         +-> PCA (2 components)
```

- **Calibration weights** is a Linear Solve node. A is the Sensors table's
  `s1, s2, s3` and b is the Meter table, linked into its second input. The
  two tables line up row by row.
- **PCA (2 components)** centres the three sensors and keeps their two
  widest directions (`pc_0`, `pc_1`).

## Run it

1. Open the graph (File > Open, or `cyxwiz-engine <project> examples/cyxgraph/linear_algebra/sensor_calibration.cyxgraph`).
2. Press **Execute Pipeline** (the green arrow in the Studio toolbar).
3. In Data Studio, click **Calibration weights** under Node results and run
   the query.

Expected result:

| column | x_meter |
| --- | --- |
| s1 | 2.0006 |
| s2 | −0.4993 |
| s3 | 1.4998 |

`numpy.linalg.lstsq` gives the same weights on these files. PCA (2
components) has 200 rows. The covariance's variances are 0.9937, 0.9150 and
0.6291, so the two components keep 75 % of the spread.

## The same from the window

Tools > Linear Algebra > **Least squares**, with A **From a table** (Sensors;
tick `s1, s2, s3`) and b **From a table** (Meter), gives
x = (2.0006, −0.4993, 1.4998). Its **Keep as a node: Linear Solve** button
adds the node this graph has. SVD of Sensors' covariance gives the PCA
directions, and **Add as a step: PCA** adds that step.

The same calibration from Python, with pycyxwiz.linalg checked against
numpy, is `examples/python/linear_algebra_calibration.py`.
