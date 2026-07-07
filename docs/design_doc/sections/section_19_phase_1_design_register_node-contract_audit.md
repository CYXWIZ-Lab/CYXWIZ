## 19) Phase 1 design register (node-contract audit)

Phase 1 objective:
- classify each editor node by runtime posture and required compatibility gates,
- establish upgrade priority,
- remove ambiguity for launch blockers.

ASCII status grammar:
- `S` = stable supported in this path
- `B` = blocked/fail-closed in this path
- `L` = legacy compatibility only
- `U` = unknown/needs verification

Legend mapping:
- `editor`: present in `NodeType` catalog.
- `compile`: accepted by `GraphCompiler` path extraction checks.
- `exec`: supported by training/runtime executors.
- `mat`: supported by pipeline materializer/runtime operator mapping.

```text
Family              Node(s)                        editor compile exec mat
------------------- ------------------------------ ----- ------- ---- ----
Layers              Dense                          Y     S       S    S
                    Dropout                        Y     S       S    U
                    BatchNorm                      Y     S       S    U
                    LayerNorm                      Y     S       B    U
                    GroupNorm                      Y     S       B    U
                    InstanceNorm                   Y     S       B    U
                    Conv1D                         Y     B       B    U
                    Conv2D                         Y     B       U    U
                    Conv3D                         Y     B       U    U
                    DepthwiseConv2D                Y     B       U    U
                    ConvTranspose2D                Y     B       B    U
                    MaxPool2D                      Y     B       U    U
                    AvgPool2D                      Y     B       U    U
                    GlobalMaxPool                  Y     B       U    U
                    GlobalAvgPool                  Y     B       U    U
                    AdaptiveAvgPool                Y     U       U    U

Activations         ReLU                           Y     S       S    U
                    Sigmoid                        Y     S       U    U
                    Tanh                           Y     S       U    U
                    Softmax                        Y     S       U    U
                    LeakyReLU                      Y     U       U    U
                    PReLU                          Y     U       U    U
                    GELU                           Y     U       U    U
                    Swish                          Y     U       U    U
                    Mish                           Y     U       U    U
                    ELU                            Y     U       U    U

Shape ops           Flatten                        Y     S       S    U
                    Reshape                        Y     S       S    U
                    View                           Y     U       U    U
                    Permute                        Y     U       U    U
                    Squeeze / Unsqueeze / Split     Y     U       U    U

Merge/Math         Add / Multiply / Average        Y     B       B    B
                    Tensor* (sum, mean, max, etc.)  Y     U/B    B    B
                    Constant / Lambda              Y     L       U    B
                    Identity                        Y    S?      U    B
                    TensorBroadcastTo / Expand       Y     U       U    B
                    TensorIndexSelect / Dot / MM     Y     U       U    B

Loss                MSELoss                        Y     S       S    U
                    CrossEntropyLoss               Y     S       S    U
                    BCELoss                        Y     S       S    U
                    BCEWithLogits                  Y     S       S    U
                    L1Loss / SmoothL1Loss / Huber   Y     S       U    U
                    NLLLoss                        Y     S       U    U
                    FocalLoss                      Y     U       U    U
                    SoftDiceLoss / Tversky / JaccardY     U       U    U

Optimizers          SGD / Adam / AdamW              Y    S       S    U
                    RMSprop / Adagrad / NAdam       Y    S       U    U

Schedulers          StepLR / CosineAnnealing / ...   Y     B       U    U

Regularization      L1Regularization                Y    B       U    U
                    L2Regularization                Y    B       U    U
                    ElasticNet                      Y    B       U    U

Data I/O            DataInput / DataOutput /        Y    S       S    U
                    DataConvert / DeployToNode...    Y    S/L     U    B
                    DatasetInput                    Y    U/L     S    S

Table source I/O    CSVFile / SQLQuery / HDF5...     Y    L       U    U
                    ArrowFile types                 Y    U       U    B/U
                    StreamingDataset                Y    L/B     U    B

Dataset family      ImageFolder / MNIST / CIFAR...    Y    L/B     U    B
                    AudioFolderDataset              Y    B       U    B
                    TextCorpusDataset               Y    B       U    B

Preprocess (core)   Normalize / OneHotEncode         Y     B       U    B
                    StandardScaler / MinMaxScaler    Y    S       U    S
                    RobustScaler / LabelEncoder      Y    S/U     U    S
                    OrdinalEncoder / TargetEncoder   Y    S/U     U    B/U
                    Binning / PolynomialFeatures    Y    U       U    U
                    OutlierDetector                 Y     U       U    S

Text                TextClean / TextTokenizer        Y    S       U    S
                    TextPadding / TFIDF/Count...    Y    S/U     U    S
                    WordEmbeddings / NER etc        Y    U       U    B/U

Sequence            TimeSeriesWindow                 Y    S       S    S
                    TimeSeriesFeatures / Lag         Y    U       U    S
                    Differencing                    Y    S       U    S
                    TimeSeriesSplit                 Y     U       U    U
                    TSeries decomposition chain      Y     U       U    U

Audio               AudioInput                      Y     B       U    B
                    Spectrogram / MelSpectrogram     Y     B       U    B
                    MFCC                            Y     B       U    B

Vision              Resize / Crop / flip / jitter    Y    S       S    S
                    ColorJitter / GaussianBlur      Y    S       U    U

Inference           DNNModelLoad / DNNDetect       Y    B       U    B
                    DNNClassify / Pose / Face       Y    U       U    B
                    PretrainedYOLO / MobileNet /    Y    B       U    B
                    OpenPose / FaceNet

Evaluation          ConfusionMatrix / ROC / PR       Y    L/B     U    B
                    LearningCurves / FeatureImportance/ Y B       B    U
                    CrossValidation

ML algorithms       KMeans / DBSCAN / GMM            Y    U/S     U/S  S
                    PCA / TSNE / UMAP               Y    S       U    S/B
                    Tree/Forest/GradientBoosting    Y    U       U    S
                    SVM / KNN / Logistic/Linear     Y    U       U    B

Clustering/etc      GMM / TSNE / UMAP              Y    U       U    S/B
                    Regression classics              Y    U       U    S

Graph control       Subgraph                        Y    L       U    U
                    Lambda / Parameter              Y    S/L     S    S
                    SignalScope / Signal nodes       Y    L/B     U    B

RL                  GymEnvironment / ReplayBuffer    Y    L/B     U    B
                    PolicyNetwork / ValueNetwork     Y    B       U    B
                    RLTraining                      Y    B       U    B
```

Status notes:
- This matrix is conservative: when compiler path is ambiguous it is intentionally marked `U` to force explicit verification before implementation assumptions.
- `S` and `U` in compile/exec/mat columns should be validated per version; many rows come from capability tables and explicit blocker lists.

### 19.1 Prioritized audit actions (Phase 1 backlog)
1. Verify every `S` row has a direct code-path test or launcher path example.
2. Replace all `U` entries with concrete state from:
   - `pipeline_runtime_capabilities.*`
   - `graph_compiler.*`
   - `training_executor.*`
   - `pipeline_materializer.*`
3. Convert `B` rows into explicit deprecation notes or enablement tasks.
4. Normalize alias-driven nodes so compile logs always show canonical and effective node type.

### 19.2 Launch-critical U normalization register

For release-readiness, unresolved `U` entries in launch-critical families are normalized here:

```text
Family/Node                    Compile  Exec  Mat   Resolution
-----------------------------  -------- ------ ----- -------------------------------
LeakyReLU / PReLU / GELU      B        B     B     No executor binding in launch-critical surface
Swish / Mish / ELU            B        B     B     Not supported for deterministic launch
Sigmoid / Softmax             B        B     B     Executor capability missing in launch path
Conv2D / Conv3D / Depthwise   B        B     B     Unsupported for current compile/exec path
ConvTranspose2D                B        B     B     Unsupported for current compile/exec path
AdaptiveAvgPool / AvgPool      B        B     B     Pooling family blocked for strict parity phase
GlobalMaxPool                  B        B     B     Pooling family blocked for strict parity phase
View / Permute                B        B     B     Operator role not normalized in active materializer path
Tensor* reduction ops          B        B     B     Backend support not proven for launch
Constant / Lambda              B        B     B     Compatibility-only nodes; blocked in launch
RMSprop / Adagrad / NAdam      B        B     B     Optimizer parity not yet stabilized
Schedulers family              B        B     B     Scheduler execution deferred
Regularization family           B        B     B     Conservatively blocked pending explicit mapping
StreamingDataset               B        B     B     Unsupported by current materializer path
AudioFolderDataset             B        B     B     Unsupported in launch envelope
TextCorpusDataset              B        B     B     Unsupported in launch envelope
TextPadding / TFIDF / Count    B        B     B     Launch support pending explicit mapping
KMeans / DBSCAN / GMM          B        B     B     Non-training family, blocked unless enabled
SVM / KNN / Logistic / Linear  B       B     B     Non-training family, blocked unless enabled
GymEnvironment / ReplayBuffer   B        B     B     RL domain excluded from current training launch
RLTraining                     B        B     B     RL domain excluded from current training launch
```

### 19.3 Known high-risk families (to stabilize first)
- Sequential blockers (already known): Conv/pooling-family and advanced schedulers.
- Fail-closed groups used for safety: SVM/KNN/legacy RL pipeline nodes with no stable training binding.
- Text/NLP and table-analysis clusters with partial materializer support.
