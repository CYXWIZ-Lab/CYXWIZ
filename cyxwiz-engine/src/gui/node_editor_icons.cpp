// Node Editor Icon Module
// Extracted from node_editor_nodes.cpp for better code organization
// Contains GetNodeIcon implementation for all icon packs

#include "node_editor.h"
#include "icons.h"
#include "IconsTabler.h"
#include "IconsRemix.h"
#include "IconsLucide.h"
#include "IconsIconoir.h"
#include "IconsPhosphor.h"

namespace gui {

// ========== Icon Implementation ==========
const char* NodeEditor::GetNodeIcon(NodeType type) {
    // Check if Tabler icon pack is selected
    if (icon_pack_ == IconPack::Tabler) {
        switch (type) {
            // Data Pipeline
            case NodeType::DatasetInput:
                return ICON_TI_DATABASE;
            case NodeType::DataLoader:
                return ICON_TI_LOADER;
            case NodeType::Augmentation:
                return ICON_TI_WAND;
            case NodeType::DataSplit:
                return ICON_TI_SITEMAP;
            case NodeType::Normalize:
                return ICON_TI_CHART_BAR;
            case NodeType::OneHotEncode:
                return ICON_TI_GRID_DOTS;

            // Core Layers
            case NodeType::Dense:
                return ICON_TI_BRAIN;
            case NodeType::Flatten:
                return ICON_TI_LAYERS_SUBTRACT;

            // Convolutional Layers
            case NodeType::Conv1D:
            case NodeType::Conv2D:
            case NodeType::Conv3D:
            case NodeType::DepthwiseConv2D:
                return ICON_TI_TABLE;

            // Pooling Layers
            case NodeType::MaxPool2D:
            case NodeType::AvgPool2D:
            case NodeType::GlobalMaxPool:
            case NodeType::GlobalAvgPool:
            case NodeType::AdaptiveAvgPool:
                return ICON_TI_ARROWS_MINIMIZE;

            // Normalization Layers
            case NodeType::BatchNorm:
            case NodeType::LayerNorm:
            case NodeType::GroupNorm:
            case NodeType::InstanceNorm:
                return ICON_TI_CHART_BAR;

            // Regularization
            case NodeType::Dropout:
                return ICON_TI_DICE;

            // Recurrent Layers
            case NodeType::RNN:
            case NodeType::LSTM:
            case NodeType::GRU:
            case NodeType::Bidirectional:
            case NodeType::TimeDistributed:
                return ICON_TI_ROTATE;
            case NodeType::Embedding:
                return ICON_TI_TYPOGRAPHY;

            // Attention & Transformer
            case NodeType::MultiHeadAttention:
            case NodeType::SelfAttention:
            case NodeType::CrossAttention:
            case NodeType::LinearAttention:
                return ICON_TI_FOCUS;
            case NodeType::TransformerEncoder:
            case NodeType::TransformerDecoder:
                return ICON_TI_CPU;
            case NodeType::PositionalEncoding:
                return ICON_TI_HASH;

            // Activation Functions
            case NodeType::ReLU:
            case NodeType::LeakyReLU:
            case NodeType::PReLU:
            case NodeType::ELU:
            case NodeType::SELU:
            case NodeType::GELU:
            case NodeType::Swish:
            case NodeType::Mish:
                return ICON_TI_BOLT;
            case NodeType::Sigmoid:
                return ICON_TI_ACTIVITY;
            case NodeType::Tanh:
                return ICON_TI_WAVE_SQUARE;
            case NodeType::Softmax:
                return ICON_TI_CHART_PIE;

            // Shape Operations
            case NodeType::Reshape:
            case NodeType::TensorReshape:
            case NodeType::View:
                return ICON_TI_ARROWS_MAXIMIZE;
            case NodeType::Permute:
                return ICON_TI_ARROWS_SHUFFLE;
            case NodeType::Squeeze:
            case NodeType::Unsqueeze:
                return ICON_TI_ARROWS_DIAGONAL_MINIMIZE;
            case NodeType::Split:
                return ICON_TI_SITEMAP;

            // Merge Operations
            case NodeType::Concatenate:
                return ICON_TI_LINK;
            case NodeType::Add:
                return ICON_TI_PLUS;
            case NodeType::Multiply:
                return ICON_TI_X;
            case NodeType::Average:
                return ICON_TI_CHART_LINE;

            // Output
            case NodeType::Output:
                return ICON_TI_CIRCLE_CHECK;

            // Loss Functions
            case NodeType::MSELoss:
            case NodeType::L1Loss:
            case NodeType::SmoothL1Loss:
            case NodeType::HuberLoss:
                return ICON_TI_CALCULATOR;
            case NodeType::CrossEntropyLoss:
            case NodeType::BCELoss:
            case NodeType::BCEWithLogits:
            case NodeType::NLLLoss:
                return ICON_TI_SCALE;

            // Optimizers
            case NodeType::SGD:
            case NodeType::Adam:
            case NodeType::AdamW:
            case NodeType::RMSprop:
            case NodeType::Adagrad:
            case NodeType::NAdam:
                return ICON_TI_ADJUSTMENTS;

            // Learning Rate Schedulers
            case NodeType::StepLR:
            case NodeType::CosineAnnealing:
            case NodeType::ReduceOnPlateau:
            case NodeType::ExponentialLR:
            case NodeType::WarmupScheduler:
                return ICON_TI_CHART_LINE;

            // Regularization Nodes
            case NodeType::L1Regularization:
            case NodeType::L2Regularization:
            case NodeType::ElasticNet:
                return ICON_TI_FILTER;

            // Utility Nodes
            case NodeType::Lambda:
                return ICON_TI_CODE;
            case NodeType::Identity:
                return ICON_TI_EQUAL;
            case NodeType::Constant:
                return ICON_TI_CIRCLE;
            case NodeType::Parameter:
                return ICON_TI_SETTINGS;

            // Signal / Control
            case NodeType::SignalSlider:
                return ICON_TI_ADJUSTMENTS_ALT;
            case NodeType::SineWave:
            case NodeType::StepSignal:
                return ICON_TI_WAVE_SINE;
            case NodeType::RampSignal:
                return ICON_TI_ARROW_UP_RIGHT;
            case NodeType::SignalScope:
                return ICON_TI_CHART_LINE;

            // Subgraph
            case NodeType::Subgraph:
                return ICON_TI_RECTANGLE_GROUP;

            // DNN Inference Nodes
            case NodeType::DNNModelLoad:
                return ICON_TI_BRAIN;
            case NodeType::DNNDetect:
            case NodeType::DNNFaceDetect:
                return ICON_TI_SCAN;
            case NodeType::DNNClassify:
                return ICON_TI_TAG;
            case NodeType::DNNPoseEstimate:
                return ICON_TI_USER;
            case NodeType::DNNPreprocess:
                return ICON_TI_WAND;
            case NodeType::PretrainedYOLO:
            case NodeType::PretrainedMobileNet:
            case NodeType::PretrainedOpenPose:
            case NodeType::PretrainedFaceNet:
                return ICON_TI_BRAIN;
            case NodeType::NonMaxSuppression:
            case NodeType::ThresholdFilter:
                return ICON_TI_FILTER;
            case NodeType::ArgMax:
            case NodeType::TopK:
                return ICON_TI_ARROW_UP_RIGHT;

            // Text Processing
            case NodeType::TextTokenizer:
                return ICON_TI_TYPOGRAPHY;
            case NodeType::TextVocabulary:
                return ICON_TI_BOOK;
            case NodeType::TextPadding:
                return ICON_TI_ALIGN_LEFT;

            // Upsampling
            case NodeType::ConvTranspose2D:
            case NodeType::Upsample:
            case NodeType::PixelShuffle:
                return ICON_TI_ARROWS_MAXIMIZE;

            // Time-Series
            case NodeType::TimeSeriesWindow:
            case NodeType::TimeSeriesFeatures:
            case NodeType::TimeSeriesSplit:
                return ICON_TI_CHART_LINE;

            // Audio
            case NodeType::AudioInput:
            case NodeType::AudioAugmentation:
                return ICON_TI_MUSIC;
            case NodeType::Spectrogram:
            case NodeType::MelSpectrogram:
            case NodeType::MFCC:
                return ICON_TI_WAVE_SQUARE;

            // RL
            case NodeType::GymEnvironment:
                return ICON_TI_FOCUS;
            case NodeType::ReplayBufferNode:
                return ICON_TI_DATABASE;
            case NodeType::PolicyNetwork:
            case NodeType::ValueNetwork:
                return ICON_TI_BRAIN;
            case NodeType::RLTraining:
                return ICON_TI_SCAN;

            // Smart I/O Nodes
            case NodeType::DataInput:
                return ICON_TI_FILE_IMPORT;
            case NodeType::DataOutput:
                return ICON_TI_FILE_EXPORT;

            // Legacy Data Source Nodes (File Formats)
            case NodeType::CSVFile:
            case NodeType::TSVFile:
            case NodeType::ExcelFile:
                return ICON_TI_FILE_SPREADSHEET;
            case NodeType::SQLQuery:
            case NodeType::HDF5Dataset:
                return ICON_TI_FILE_DATABASE;
            case NodeType::ParquetFile:
            case NodeType::FeatherFile:
            case NodeType::ArrowIPCFile:
            case NodeType::NumPyFile:
                return ICON_TI_FILE_ANALYTICS;
            case NodeType::JSONFile:
            case NodeType::ARFFFile:
                return ICON_TI_FILE_CODE;
            case NodeType::TXTFile:
            case NodeType::TextCorpusDataset:
                return ICON_TI_FILE_TEXT;
            case NodeType::RESTAPISource:
                return ICON_TI_API;
            case NodeType::ImageCSVDataset:
            case NodeType::ImageFolderDataset:
            case NodeType::FashionMNISTDataset:
            case NodeType::CIFAR100Dataset:
                return ICON_TI_PHOTO;
            case NodeType::StreamingDataset:
            case NodeType::TimeSeriesCSV:
            case NodeType::AudioFolderDataset:
                return ICON_TI_DATABASE;

            // Data Transform Nodes
            case NodeType::FilterRows:
                return ICON_TI_FILTER_EDIT;
            case NodeType::SelectColumns:
                return ICON_TI_COLUMNS;
            case NodeType::JoinTables:
                return ICON_TI_ARROW_MERGE;
            case NodeType::GroupByAggregate:
                return ICON_TI_HIERARCHY;
            case NodeType::SortRows:
                return ICON_TI_SORT_ASCENDING;
            case NodeType::FillMissingValues:
                return ICON_TI_TRANSFORM;
            case NodeType::RemoveDuplicateRows:
                return ICON_TI_TRASH;
            case NodeType::PivotTable:
            case NodeType::Unpivot:
                return ICON_TI_SWITCH;
            case NodeType::UnionTables:
            case NodeType::RowAppender:
                return ICON_TI_STACK;
            case NodeType::RenameColumns:
                return ICON_TI_REPLACE;

            // Analytics Nodes
            case NodeType::DescribeStats:
                return ICON_TI_REPORT_ANALYTICS;
            case NodeType::VisualizeData:
                return ICON_TI_CHART_DOTS;
            case NodeType::SampleRows:
                return ICON_TI_DICE;
            case NodeType::CorrelationMatrix:
                return ICON_TI_CHART_RADAR;
            case NodeType::ValueCounts:
                return ICON_TI_PERCENTAGE;
            case NodeType::CrossTabulation:
                return ICON_TI_GRID_3X3;

            // Data Export Nodes
            case NodeType::ExportCSV:
            case NodeType::ExportExcel:
                return ICON_TI_FILE_SPREADSHEET;
            case NodeType::ExportParquet:
                return ICON_TI_FILE_ANALYTICS;
            case NodeType::ExportSQL:
                return ICON_TI_FILE_DATABASE;
            case NodeType::ExportJSON:
                return ICON_TI_FILE_CODE;

            // KNIME-Style Table Manipulation Nodes
            case NodeType::RowToColumnNames:
                return ICON_TI_LAYOUT_ROWS;
            case NodeType::TableSplitter:
                return ICON_TI_ARROWS_SPLIT;
            case NodeType::CellExtractor:
            case NodeType::CellUpdater:
                return ICON_TI_CELL;
            case NodeType::TableCropper:
                return ICON_TI_CROP;
            case NodeType::ColumnAppender:
                return ICON_TI_LAYOUT_COLUMNS;
            case NodeType::StringManipulation:
                return ICON_TI_TEXT_WRAP;
            case NodeType::MathFormula:
                return ICON_TI_MATH_FUNCTION;
            case NodeType::RuleEngine:
                return ICON_TI_RULE;

            // Machine Learning Algorithms - Clustering
            case NodeType::KMeansCluster:
                return ICON_TI_CIRCLES;
            case NodeType::DBSCANCluster:
                return ICON_TI_AFFILIATE;
            case NodeType::HierarchicalCluster:
                return ICON_TI_TIMELINE;
            case NodeType::GMMCluster:
                return ICON_TI_CHART_RADAR;

            // Machine Learning - Dimensionality Reduction
            case NodeType::PCANode:
            case NodeType::TSNENode:
            case NodeType::UMAPNode:
                return ICON_TI_3D_CUBE_SPHERE;

            // Machine Learning - Classification
            case NodeType::DecisionTreeClassifier:
                return ICON_TI_BINARY_TREE;
            case NodeType::RandomForestClassifier:
                return ICON_TI_PLANT;
            case NodeType::GradientBoostingClassifier:
                return ICON_TI_CHART_CANDLE;
            case NodeType::TreeModelPredictor:
                return ICON_TI_FILE_ANALYTICS;
            case NodeType::SVMClassifier:
            case NodeType::SVMRegressor:
                return ICON_TI_VECTOR;
            case NodeType::KNNClassifier:
                return ICON_TI_TARGET;
            case NodeType::NaiveBayesClassifier:
                return ICON_TI_CHART_PIE;
            case NodeType::LogisticRegressionNode:
            case NodeType::LinearRegressionNode:
            case NodeType::PolynomialRegressionNode:
                return ICON_TI_CHART_LINE;

            // Model Evaluation Nodes
            case NodeType::ConfusionMatrixNode:
                return ICON_TI_CHECKBOX;
            case NodeType::ROCCurveNode:
            case NodeType::PRCurveNode:
                return ICON_TI_CHART_AREA;
            case NodeType::LearningCurvesNode:
                return ICON_TI_TRENDING_UP;
            case NodeType::FeatureImportanceNode:
                return ICON_TI_CHART_BAR;
            case NodeType::CrossValidationNode:
                return ICON_TI_COPY;
            case NodeType::RegressionMetricsNode:
                return ICON_TI_LIST_NUMBERS;

            // Data Preprocessing Nodes
            case NodeType::StandardScaler:
            case NodeType::MinMaxScaler:
            case NodeType::RobustScaler:
                return ICON_TI_RULER;
            case NodeType::LabelEncoder:
            case NodeType::OrdinalEncoder:
            case NodeType::TargetEncoder:
                return ICON_TI_ABC;

            // Advanced Preprocessing
            case NodeType::OutlierDetector:
                return ICON_TI_BUG;
            case NodeType::ImagePreprocessor:
            case NodeType::QualityAnalyzer:
                return ICON_TI_PHOTO;
            case NodeType::DataValidator:
                return ICON_TI_CHECKBOX;

            case NodeType::PluginCustom:
                return ICON_TI_PLUG;

            default:
                return ICON_TI_CIRCLE_NODES;
        }
    }

    // Remix Icon pack
    if (icon_pack_ == IconPack::Remix) {
        switch (type) {
            // Data Pipeline
            case NodeType::DatasetInput:
            case NodeType::DataLoader:
                return ICON_RI_DATABASE;
            case NodeType::Augmentation:
                return ICON_RI_SPARKLING;
            case NodeType::DataSplit:
                return ICON_RI_NODE_TREE;
            case NodeType::Normalize:
            case NodeType::BatchNorm:
            case NodeType::LayerNorm:
            case NodeType::GroupNorm:
            case NodeType::InstanceNorm:
                return ICON_RI_BAR_CHART;

            // Core Layers
            case NodeType::Dense:
            case NodeType::DNNModelLoad:
            case NodeType::PolicyNetwork:
            case NodeType::ValueNetwork:
            case NodeType::PretrainedYOLO:
            case NodeType::PretrainedMobileNet:
            case NodeType::PretrainedOpenPose:
            case NodeType::PretrainedFaceNet:
                return ICON_RI_BRAIN;
            case NodeType::Flatten:
            case NodeType::GroupByAggregate:
            case NodeType::UnionTables:
            case NodeType::RowAppender:
                return ICON_RI_STACK;

            // Convolutional Layers
            case NodeType::Conv1D:
            case NodeType::Conv2D:
            case NodeType::Conv3D:
            case NodeType::DepthwiseConv2D:
            case NodeType::PivotTable:
            case NodeType::Unpivot:
            case NodeType::CorrelationMatrix:
                return ICON_RI_GRID;

            // Pooling Layers
            case NodeType::MaxPool2D:
            case NodeType::AvgPool2D:
            case NodeType::GlobalMaxPool:
            case NodeType::GlobalAvgPool:
            case NodeType::AdaptiveAvgPool:
            case NodeType::Squeeze:
            case NodeType::Unsqueeze:
            case NodeType::TableCropper:
            case NodeType::PCANode:
            case NodeType::TSNENode:
            case NodeType::UMAPNode:
                return ICON_RI_BOX;

            // Regularization
            case NodeType::Dropout:
            case NodeType::SampleRows:
                return ICON_RI_SCATTER;

            // Recurrent Layers
            case NodeType::RNN:
            case NodeType::LSTM:
            case NodeType::GRU:
            case NodeType::Bidirectional:
            case NodeType::TimeDistributed:
                return ICON_RI_REFRESH;
            case NodeType::Embedding:
            case NodeType::TextTokenizer:
            case NodeType::StringManipulation:
                return ICON_RI_TEXT;

            // Attention & Transformer
            case NodeType::MultiHeadAttention:
            case NodeType::SelfAttention:
            case NodeType::CrossAttention:
            case NodeType::LinearAttention:
            case NodeType::GymEnvironment:
            case NodeType::KNNClassifier:
                return ICON_RI_TARGET;
            case NodeType::TransformerEncoder:
            case NodeType::TransformerDecoder:
                return ICON_RI_CPU;
            case NodeType::PositionalEncoding:
            case NodeType::ValueCounts:
                return ICON_RI_BOLD;

            // Activation Functions
            case NodeType::ReLU:
            case NodeType::LeakyReLU:
            case NodeType::PReLU:
            case NodeType::ELU:
            case NodeType::SELU:
            case NodeType::GELU:
            case NodeType::Swish:
            case NodeType::Mish:
                return ICON_RI_FLASHLIGHT;
            case NodeType::Sigmoid:
            case NodeType::Spectrogram:
            case NodeType::MelSpectrogram:
            case NodeType::MFCC:
            case NodeType::SineWave:
            case NodeType::StepSignal:
            case NodeType::Tanh:
                return ICON_RI_PULSE;
            case NodeType::Softmax:
            case NodeType::GMMCluster:
            case NodeType::NaiveBayesClassifier:
                return ICON_RI_PIE_CHART;

            // Shape Operations
            case NodeType::Reshape:
            case NodeType::TensorReshape:
            case NodeType::View:
            case NodeType::ConvTranspose2D:
            case NodeType::Upsample:
            case NodeType::PixelShuffle:
                return ICON_RI_BOX;
            case NodeType::Permute:
                return ICON_RI_SHUFFLE;
            case NodeType::Split:
            case NodeType::TableSplitter:
            case NodeType::HierarchicalCluster:
            case NodeType::DecisionTreeClassifier:
            case NodeType::TreeModelPredictor:
                return ICON_RI_NODE_TREE;

            // Merge Operations
            case NodeType::Concatenate:
            case NodeType::JoinTables:
                return ICON_RI_MERGE;
            case NodeType::Add:
                return ICON_RI_ADD;
            case NodeType::Multiply:
                return ICON_RI_CLOSE;
            case NodeType::Average:
            case NodeType::TimeSeriesWindow:
            case NodeType::TimeSeriesFeatures:
            case NodeType::TimeSeriesSplit:
            case NodeType::SignalScope:
            case NodeType::VisualizeData:
            case NodeType::LearningCurvesNode:
            case NodeType::ROCCurveNode:
            case NodeType::PRCurveNode:
            case NodeType::GradientBoostingClassifier:
            case NodeType::LogisticRegressionNode:
            case NodeType::LinearRegressionNode:
            case NodeType::PolynomialRegressionNode:
                return ICON_RI_LINE_CHART;

            // Output
            case NodeType::Output:
            case NodeType::DataValidator:
                return ICON_RI_CHECK;

            // Loss Functions
            case NodeType::MSELoss:
            case NodeType::L1Loss:
            case NodeType::SmoothL1Loss:
            case NodeType::HuberLoss:
            case NodeType::MathFormula:
            case NodeType::RegressionMetricsNode:
                return ICON_RI_CALCULATOR;
            case NodeType::CrossEntropyLoss:
            case NodeType::BCELoss:
            case NodeType::BCEWithLogits:
            case NodeType::NLLLoss:
                return ICON_RI_FUNCTION;

            // Optimizers
            case NodeType::SGD:
            case NodeType::Adam:
            case NodeType::AdamW:
            case NodeType::RMSprop:
            case NodeType::Adagrad:
            case NodeType::NAdam:
            case NodeType::StandardScaler:
            case NodeType::MinMaxScaler:
            case NodeType::RobustScaler:
            case NodeType::SignalSlider:
                return ICON_RI_SETTINGS;

            // Learning Rate Schedulers
            case NodeType::StepLR:
            case NodeType::CosineAnnealing:
            case NodeType::ReduceOnPlateau:
            case NodeType::ExponentialLR:
            case NodeType::WarmupScheduler:
                return ICON_RI_LINE_CHART;

            // Regularization Nodes
            case NodeType::L1Regularization:
            case NodeType::L2Regularization:
            case NodeType::ElasticNet:
            case NodeType::FilterRows:
            case NodeType::NonMaxSuppression:
            case NodeType::ThresholdFilter:
                return ICON_RI_FILTER;

            // Utility Nodes
            case NodeType::Lambda:
            case NodeType::RuleEngine:
                return ICON_RI_CODE;
            case NodeType::Identity:
                return ICON_RI_SUBTRACT;
            case NodeType::Constant:
            case NodeType::CellExtractor:
            case NodeType::CellUpdater:
            case NodeType::SVMClassifier:
            case NodeType::SVMRegressor:
                return ICON_RI_SQUARE;
            case NodeType::Parameter:
                return ICON_RI_SETTINGS;

            // Signal / Control
            case NodeType::RampSignal:
            case NodeType::ArgMax:
            case NodeType::TopK:
                return ICON_RI_ARROW_UP;

            // Subgraph
            case NodeType::Subgraph:
                return ICON_RI_APPS;

            // DNN Inference Nodes
            case NodeType::DNNDetect:
            case NodeType::DNNFaceDetect:
            case NodeType::RLTraining:
                return ICON_RI_SEARCH;
            case NodeType::DNNClassify:
            case NodeType::LabelEncoder:
            case NodeType::OrdinalEncoder:
            case NodeType::TargetEncoder:
                return ICON_RI_BOLD;
            case NodeType::DNNPoseEstimate:
                return ICON_RI_ROBOT;
            case NodeType::DNNPreprocess:
                return ICON_RI_SPARKLING;

            // Text Processing
            case NodeType::TextVocabulary:
                return ICON_RI_FILE_LIST;
            case NodeType::TextPadding:
                return ICON_RI_TEXT;

            // Audio
            case NodeType::AudioInput:
            case NodeType::AudioAugmentation:
                return ICON_RI_MUSIC;

            // RL
            case NodeType::ReplayBufferNode:
            case NodeType::SQLQuery:
            case NodeType::HDF5Dataset:
            case NodeType::StreamingDataset:
            case NodeType::TimeSeriesCSV:
            case NodeType::AudioFolderDataset:
            case NodeType::ExportSQL:
                return ICON_RI_DATABASE;

            // Smart I/O Nodes
            case NodeType::DataInput:
                return ICON_RI_IMPORT;
            case NodeType::DataOutput:
                return ICON_RI_EXPORT;

            // File Formats
            case NodeType::CSVFile:
            case NodeType::TSVFile:
            case NodeType::ExcelFile:
            case NodeType::ExportCSV:
            case NodeType::ExportExcel:
                return ICON_RI_FILE_EXCEL;
            case NodeType::ParquetFile:
            case NodeType::FeatherFile:
            case NodeType::ArrowIPCFile:
            case NodeType::NumPyFile:
            case NodeType::ExportParquet:
                return ICON_RI_FILE_LIST;
            case NodeType::JSONFile:
            case NodeType::ARFFFile:
            case NodeType::ExportJSON:
                return ICON_RI_FILE_CODE;
            case NodeType::TXTFile:
            case NodeType::TextCorpusDataset:
                return ICON_RI_FILE_TEXT;
            case NodeType::RESTAPISource:
                return ICON_RI_SERVER;
            case NodeType::ImageCSVDataset:
            case NodeType::ImageFolderDataset:
            case NodeType::FashionMNISTDataset:
            case NodeType::CIFAR100Dataset:
            case NodeType::ImagePreprocessor:
            case NodeType::QualityAnalyzer:
                return ICON_RI_IMAGE;

            // Data Transform Nodes
            case NodeType::SelectColumns:
            case NodeType::ColumnAppender:
            case NodeType::CrossTabulation:
            case NodeType::ConfusionMatrixNode:
                return ICON_RI_GRID;
            case NodeType::SortRows:
                return ICON_RI_SORT;
            case NodeType::FillMissingValues:
                return ICON_RI_SPARKLING;
            case NodeType::RemoveDuplicateRows:
                return ICON_RI_CUT;
            case NodeType::RenameColumns:
            case NodeType::RowToColumnNames:
                return ICON_RI_EDIT;

            // Analytics Nodes
            case NodeType::DescribeStats:
            case NodeType::FeatureImportanceNode:
                return ICON_RI_BAR_CHART;

            // ML Algorithms
            case NodeType::KMeansCluster:
            case NodeType::DBSCANCluster:
                return ICON_RI_CIRCLE;
            case NodeType::RandomForestClassifier:
                return ICON_RI_STACK;
            case NodeType::CrossValidationNode:
                return ICON_RI_FILE_COPY;
            case NodeType::OutlierDetector:
                return ICON_RI_BUG;
            case NodeType::OneHotEncode:
                return ICON_RI_GRID;

            case NodeType::PluginCustom:
                return ICON_RI_PLUGIN;

            default:
                return ICON_RI_APPS;
        }
    }

    // Lucide Icon pack
    if (icon_pack_ == IconPack::Lucide) {
        switch (type) {
            // Data Pipeline
            case NodeType::DatasetInput:
            case NodeType::DataLoader:
            case NodeType::ReplayBufferNode:
            case NodeType::SQLQuery:
            case NodeType::HDF5Dataset:
            case NodeType::StreamingDataset:
            case NodeType::TimeSeriesCSV:
            case NodeType::AudioFolderDataset:
            case NodeType::ExportSQL:
                return ICON_LU_DATABASE;
            case NodeType::Augmentation:
            case NodeType::DNNPreprocess:
            case NodeType::FillMissingValues:
                return ICON_LU_SPARKLES;
            case NodeType::DataSplit:
            case NodeType::Split:
            case NodeType::TableSplitter:
            case NodeType::HierarchicalCluster:
            case NodeType::DecisionTreeClassifier:
                return ICON_LU_SITEMAP;
            case NodeType::Normalize:
            case NodeType::BatchNorm:
            case NodeType::LayerNorm:
            case NodeType::GroupNorm:
            case NodeType::InstanceNorm:
            case NodeType::DescribeStats:
            case NodeType::FeatureImportanceNode:
                return ICON_LU_BAR_CHART;

            // Core Layers
            case NodeType::Dense:
            case NodeType::DNNModelLoad:
            case NodeType::PolicyNetwork:
            case NodeType::ValueNetwork:
            case NodeType::PretrainedYOLO:
            case NodeType::PretrainedMobileNet:
            case NodeType::PretrainedOpenPose:
            case NodeType::PretrainedFaceNet:
                return ICON_LU_BRAIN;
            case NodeType::Flatten:
            case NodeType::GroupByAggregate:
            case NodeType::UnionTables:
            case NodeType::RowAppender:
            case NodeType::RandomForestClassifier:
                return ICON_LU_LAYERS;

            // Convolutional Layers
            case NodeType::Conv1D:
            case NodeType::Conv2D:
            case NodeType::Conv3D:
            case NodeType::DepthwiseConv2D:
            case NodeType::PivotTable:
            case NodeType::Unpivot:
            case NodeType::CorrelationMatrix:
            case NodeType::SelectColumns:
            case NodeType::ColumnAppender:
            case NodeType::CrossTabulation:
            case NodeType::ConfusionMatrixNode:
            case NodeType::OneHotEncode:
                return ICON_LU_GRID_3X3;

            // Pooling Layers
            case NodeType::MaxPool2D:
            case NodeType::AvgPool2D:
            case NodeType::GlobalMaxPool:
            case NodeType::GlobalAvgPool:
            case NodeType::AdaptiveAvgPool:
            case NodeType::Squeeze:
            case NodeType::Unsqueeze:
            case NodeType::TableCropper:
            case NodeType::PCANode:
            case NodeType::TSNENode:
            case NodeType::UMAPNode:
            case NodeType::Reshape:
            case NodeType::TensorReshape:
            case NodeType::View:
            case NodeType::ConvTranspose2D:
            case NodeType::Upsample:
            case NodeType::PixelShuffle:
                return ICON_LU_BOX;

            // Regularization
            case NodeType::Dropout:
            case NodeType::SampleRows:
                return ICON_LU_SCATTER_CHART;

            // Recurrent Layers
            case NodeType::RNN:
            case NodeType::LSTM:
            case NodeType::GRU:
            case NodeType::Bidirectional:
            case NodeType::TimeDistributed:
                return ICON_LU_REPEAT;
            case NodeType::Embedding:
            case NodeType::TextTokenizer:
            case NodeType::StringManipulation:
                return ICON_LU_TYPE;
            case NodeType::TextVocabulary:
                return ICON_LU_LIST;
            case NodeType::TextPadding:
                return ICON_LU_ALIGN_LEFT;

            // Attention & Transformer
            case NodeType::MultiHeadAttention:
            case NodeType::SelfAttention:
            case NodeType::CrossAttention:
            case NodeType::LinearAttention:
            case NodeType::GymEnvironment:
            case NodeType::KNNClassifier:
                return ICON_LU_TARGET;
            case NodeType::TransformerEncoder:
            case NodeType::TransformerDecoder:
                return ICON_LU_CPU;
            case NodeType::PositionalEncoding:
            case NodeType::ValueCounts:
                return ICON_LU_HASH;

            // Activation Functions
            case NodeType::ReLU:
            case NodeType::LeakyReLU:
            case NodeType::PReLU:
            case NodeType::ELU:
            case NodeType::SELU:
            case NodeType::GELU:
            case NodeType::Swish:
            case NodeType::Mish:
                return ICON_LU_ZAP;
            case NodeType::Sigmoid:
            case NodeType::Spectrogram:
            case NodeType::MelSpectrogram:
            case NodeType::MFCC:
            case NodeType::SineWave:
            case NodeType::StepSignal:
            case NodeType::Tanh:
                return ICON_LU_ACTIVITY;
            case NodeType::Softmax:
            case NodeType::GMMCluster:
            case NodeType::NaiveBayesClassifier:
                return ICON_LU_PIE_CHART;

            // Shape Operations
            case NodeType::Permute:
                return ICON_LU_SHUFFLE;

            // Merge Operations
            case NodeType::Concatenate:
            case NodeType::JoinTables:
                return ICON_LU_LINK;
            case NodeType::Add:
                return ICON_LU_PLUS;
            case NodeType::Multiply:
                return ICON_LU_X;
            case NodeType::Average:
            case NodeType::TimeSeriesWindow:
            case NodeType::TimeSeriesFeatures:
            case NodeType::TimeSeriesSplit:
            case NodeType::SignalScope:
            case NodeType::VisualizeData:
            case NodeType::LearningCurvesNode:
            case NodeType::ROCCurveNode:
            case NodeType::PRCurveNode:
            case NodeType::GradientBoostingClassifier:
            case NodeType::LogisticRegressionNode:
            case NodeType::LinearRegressionNode:
            case NodeType::PolynomialRegressionNode:
                return ICON_LU_LINE_CHART;

            // Output
            case NodeType::Output:
            case NodeType::DataValidator:
                return ICON_LU_CHECK_CIRCLE;

            // Loss Functions
            case NodeType::MSELoss:
            case NodeType::L1Loss:
            case NodeType::SmoothL1Loss:
            case NodeType::HuberLoss:
            case NodeType::MathFormula:
            case NodeType::RegressionMetricsNode:
                return ICON_LU_CALCULATOR;
            case NodeType::CrossEntropyLoss:
            case NodeType::BCELoss:
            case NodeType::BCEWithLogits:
            case NodeType::NLLLoss:
                return ICON_LU_FUNCTION_SQUARE;

            // Optimizers & Scalers
            case NodeType::SGD:
            case NodeType::Adam:
            case NodeType::AdamW:
            case NodeType::RMSprop:
            case NodeType::Adagrad:
            case NodeType::NAdam:
            case NodeType::StandardScaler:
            case NodeType::MinMaxScaler:
            case NodeType::RobustScaler:
            case NodeType::SignalSlider:
            case NodeType::Parameter:
                return ICON_LU_SETTINGS;

            // Regularization Nodes
            case NodeType::L1Regularization:
            case NodeType::L2Regularization:
            case NodeType::ElasticNet:
            case NodeType::FilterRows:
            case NodeType::NonMaxSuppression:
            case NodeType::ThresholdFilter:
                return ICON_LU_FILTER;

            // Utility Nodes
            case NodeType::Lambda:
            case NodeType::RuleEngine:
                return ICON_LU_CODE;
            case NodeType::Identity:
                return ICON_LU_EQUAL;
            case NodeType::Constant:
            case NodeType::CellExtractor:
            case NodeType::CellUpdater:
            case NodeType::SVMClassifier:
            case NodeType::SVMRegressor:
                return ICON_LU_SQUARE;

            // Signal / Control
            case NodeType::RampSignal:
            case NodeType::ArgMax:
            case NodeType::TopK:
                return ICON_LU_TRENDING_UP;

            // Subgraph
            case NodeType::Subgraph:
                return ICON_LU_WORKFLOW;

            // DNN Inference Nodes
            case NodeType::DNNDetect:
            case NodeType::DNNFaceDetect:
            case NodeType::RLTraining:
                return ICON_LU_SEARCH;
            case NodeType::DNNClassify:
            case NodeType::LabelEncoder:
            case NodeType::OrdinalEncoder:
            case NodeType::TargetEncoder:
                return ICON_LU_TAGS;
            case NodeType::DNNPoseEstimate:
                return ICON_LU_BOT;

            // Audio
            case NodeType::AudioInput:
            case NodeType::AudioAugmentation:
                return ICON_LU_MUSIC;

            // Smart I/O Nodes
            case NodeType::DataInput:
                return ICON_LU_IMPORT;
            case NodeType::DataOutput:
                return ICON_LU_DOWNLOAD;

            // File Formats
            case NodeType::CSVFile:
            case NodeType::TSVFile:
            case NodeType::ExcelFile:
            case NodeType::ExportCSV:
            case NodeType::ExportExcel:
                return ICON_LU_FILE_SPREADSHEET;
            case NodeType::ParquetFile:
            case NodeType::FeatherFile:
            case NodeType::ArrowIPCFile:
            case NodeType::NumPyFile:
            case NodeType::ExportParquet:
                return ICON_LU_FILE_TEXT;
            case NodeType::JSONFile:
            case NodeType::ARFFFile:
            case NodeType::ExportJSON:
                return ICON_LU_FILE_CODE;
            case NodeType::TXTFile:
            case NodeType::TextCorpusDataset:
                return ICON_LU_FILE_TEXT;
            case NodeType::RESTAPISource:
                return ICON_LU_SERVER;
            case NodeType::ImageCSVDataset:
            case NodeType::ImageFolderDataset:
            case NodeType::FashionMNISTDataset:
            case NodeType::CIFAR100Dataset:
            case NodeType::ImagePreprocessor:
            case NodeType::QualityAnalyzer:
                return ICON_LU_IMAGE;

            // Data Transform Nodes
            case NodeType::SortRows:
                return ICON_LU_SORT_ASC;
            case NodeType::RemoveDuplicateRows:
                return ICON_LU_TRASH;
            case NodeType::RenameColumns:
            case NodeType::RowToColumnNames:
                return ICON_LU_EDIT;

            // ML Algorithms
            case NodeType::KMeansCluster:
            case NodeType::DBSCANCluster:
                return ICON_LU_CIRCLE;
            case NodeType::CrossValidationNode:
                return ICON_LU_COPY;
            case NodeType::OutlierDetector:
                return ICON_LU_BUG;

            case NodeType::PluginCustom:
                return ICON_LU_PLUG;

            default:
                return ICON_LU_NETWORK;
        }
    }

    // Iconoir Icon pack
    if (icon_pack_ == IconPack::Iconoir) {
        switch (type) {
            // Data Pipeline
            case NodeType::DatasetInput:
            case NodeType::DataLoader:
            case NodeType::ReplayBufferNode:
            case NodeType::SQLQuery:
            case NodeType::HDF5Dataset:
            case NodeType::StreamingDataset:
            case NodeType::TimeSeriesCSV:
            case NodeType::AudioFolderDataset:
            case NodeType::ExportSQL:
                return ICON_IO_DATABASE;
            case NodeType::Augmentation:
            case NodeType::DNNPreprocess:
            case NodeType::FillMissingValues:
                return ICON_IO_SPARKS;
            case NodeType::DataSplit:
            case NodeType::Split:
            case NodeType::TableSplitter:
            case NodeType::HierarchicalCluster:
            case NodeType::DecisionTreeClassifier:
                return ICON_IO_TREE;
            case NodeType::Normalize:
            case NodeType::BatchNorm:
            case NodeType::LayerNorm:
            case NodeType::GroupNorm:
            case NodeType::InstanceNorm:
            case NodeType::DescribeStats:
            case NodeType::FeatureImportanceNode:
                return ICON_IO_STATS_UP;

            // Core Layers
            case NodeType::Dense:
            case NodeType::DNNModelLoad:
            case NodeType::PolicyNetwork:
            case NodeType::ValueNetwork:
            case NodeType::PretrainedYOLO:
            case NodeType::PretrainedMobileNet:
            case NodeType::PretrainedOpenPose:
            case NodeType::PretrainedFaceNet:
                return ICON_IO_BRAIN;
            case NodeType::Flatten:
            case NodeType::GroupByAggregate:
            case NodeType::UnionTables:
            case NodeType::RowAppender:
            case NodeType::RandomForestClassifier:
                return ICON_IO_LAYERS;

            // Convolutional Layers
            case NodeType::Conv1D:
            case NodeType::Conv2D:
            case NodeType::Conv3D:
            case NodeType::DepthwiseConv2D:
            case NodeType::PivotTable:
            case NodeType::Unpivot:
            case NodeType::CorrelationMatrix:
            case NodeType::SelectColumns:
            case NodeType::ColumnAppender:
            case NodeType::CrossTabulation:
            case NodeType::ConfusionMatrixNode:
            case NodeType::OneHotEncode:
                return ICON_IO_GRID;

            // Pooling Layers
            case NodeType::MaxPool2D:
            case NodeType::AvgPool2D:
            case NodeType::GlobalMaxPool:
            case NodeType::GlobalAvgPool:
            case NodeType::AdaptiveAvgPool:
            case NodeType::Squeeze:
            case NodeType::Unsqueeze:
            case NodeType::TableCropper:
            case NodeType::PCANode:
            case NodeType::TSNENode:
            case NodeType::UMAPNode:
            case NodeType::Reshape:
            case NodeType::TensorReshape:
            case NodeType::View:
            case NodeType::ConvTranspose2D:
            case NodeType::Upsample:
            case NodeType::PixelShuffle:
                return ICON_IO_BOX;

            // Regularization
            case NodeType::Dropout:
            case NodeType::SampleRows:
                return ICON_IO_SHUFFLE;

            // Recurrent Layers
            case NodeType::RNN:
            case NodeType::LSTM:
            case NodeType::GRU:
            case NodeType::Bidirectional:
            case NodeType::TimeDistributed:
                return ICON_IO_REFRESH;
            case NodeType::Embedding:
            case NodeType::TextTokenizer:
            case NodeType::StringManipulation:
                return ICON_IO_TEXT;
            case NodeType::TextVocabulary:
                return ICON_IO_LIST;
            case NodeType::TextPadding:
                return ICON_IO_ALIGN_LEFT;

            // Attention & Transformer
            case NodeType::MultiHeadAttention:
            case NodeType::SelfAttention:
            case NodeType::CrossAttention:
            case NodeType::LinearAttention:
            case NodeType::GymEnvironment:
            case NodeType::KNNClassifier:
                return ICON_IO_TARGET;
            case NodeType::TransformerEncoder:
            case NodeType::TransformerDecoder:
                return ICON_IO_CPU;
            case NodeType::PositionalEncoding:
            case NodeType::ValueCounts:
                return ICON_IO_HASHTAG;

            // Activation Functions
            case NodeType::ReLU:
            case NodeType::LeakyReLU:
            case NodeType::PReLU:
            case NodeType::ELU:
            case NodeType::SELU:
            case NodeType::GELU:
            case NodeType::Swish:
            case NodeType::Mish:
                return ICON_IO_FLASH;
            case NodeType::Sigmoid:
            case NodeType::Spectrogram:
            case NodeType::MelSpectrogram:
            case NodeType::MFCC:
            case NodeType::SineWave:
            case NodeType::StepSignal:
            case NodeType::Tanh:
                return ICON_IO_WAVE;
            case NodeType::Softmax:
            case NodeType::GMMCluster:
            case NodeType::NaiveBayesClassifier:
                return ICON_IO_PIE_CHART;

            // Shape Operations
            case NodeType::Permute:
                return ICON_IO_SHUFFLE;

            // Merge Operations
            case NodeType::Concatenate:
            case NodeType::JoinTables:
                return ICON_IO_LINK;
            case NodeType::Add:
                return ICON_IO_PLUS;
            case NodeType::Multiply:
                return ICON_IO_MINUS;
            case NodeType::Average:
            case NodeType::TimeSeriesWindow:
            case NodeType::TimeSeriesFeatures:
            case NodeType::TimeSeriesSplit:
            case NodeType::SignalScope:
            case NodeType::VisualizeData:
            case NodeType::LearningCurvesNode:
            case NodeType::ROCCurveNode:
            case NodeType::PRCurveNode:
            case NodeType::GradientBoostingClassifier:
            case NodeType::LogisticRegressionNode:
            case NodeType::LinearRegressionNode:
            case NodeType::PolynomialRegressionNode:
                return ICON_IO_GRAPH_UP;

            // Output
            case NodeType::Output:
            case NodeType::DataValidator:
                return ICON_IO_CHECK_CIRCLE;

            // Loss Functions
            case NodeType::MSELoss:
            case NodeType::L1Loss:
            case NodeType::SmoothL1Loss:
            case NodeType::HuberLoss:
            case NodeType::MathFormula:
            case NodeType::RegressionMetricsNode:
                return ICON_IO_CALCULATOR;
            case NodeType::CrossEntropyLoss:
            case NodeType::BCELoss:
            case NodeType::BCEWithLogits:
            case NodeType::NLLLoss:
                return ICON_IO_SIGMA;

            // Optimizers & Scalers
            case NodeType::SGD:
            case NodeType::Adam:
            case NodeType::AdamW:
            case NodeType::RMSprop:
            case NodeType::Adagrad:
            case NodeType::NAdam:
            case NodeType::StandardScaler:
            case NodeType::MinMaxScaler:
            case NodeType::RobustScaler:
            case NodeType::SignalSlider:
            case NodeType::Parameter:
                return ICON_IO_SETTINGS;

            // Regularization Nodes
            case NodeType::L1Regularization:
            case NodeType::L2Regularization:
            case NodeType::ElasticNet:
            case NodeType::FilterRows:
            case NodeType::NonMaxSuppression:
            case NodeType::ThresholdFilter:
                return ICON_IO_FILTER;

            // Utility Nodes
            case NodeType::Lambda:
            case NodeType::RuleEngine:
                return ICON_IO_CODE;
            case NodeType::Identity:
                return ICON_IO_MINUS;
            case NodeType::Constant:
            case NodeType::CellExtractor:
            case NodeType::CellUpdater:
            case NodeType::SVMClassifier:
            case NodeType::SVMRegressor:
                return ICON_IO_SQUARE;

            // Signal / Control
            case NodeType::RampSignal:
            case NodeType::ArgMax:
            case NodeType::TopK:
                return ICON_IO_ARROW_UP;

            // Subgraph
            case NodeType::Subgraph:
                return ICON_IO_APPS;

            // DNN Inference Nodes
            case NodeType::DNNDetect:
            case NodeType::DNNFaceDetect:
            case NodeType::RLTraining:
                return ICON_IO_SEARCH;
            case NodeType::DNNClassify:
            case NodeType::LabelEncoder:
            case NodeType::OrdinalEncoder:
            case NodeType::TargetEncoder:
                return ICON_IO_TAG;
            case NodeType::DNNPoseEstimate:
                return ICON_IO_CUBE;

            // Audio
            case NodeType::AudioInput:
            case NodeType::AudioAugmentation:
                return ICON_IO_MUSIC;

            // Smart I/O Nodes
            case NodeType::DataInput:
                return ICON_IO_IMPORT;
            case NodeType::DataOutput:
                return ICON_IO_DOWNLOAD;

            // File Formats
            case NodeType::CSVFile:
            case NodeType::TSVFile:
            case NodeType::ExcelFile:
            case NodeType::ExportCSV:
            case NodeType::ExportExcel:
                return ICON_IO_TABLE;
            case NodeType::ParquetFile:
            case NodeType::FeatherFile:
            case NodeType::ArrowIPCFile:
            case NodeType::NumPyFile:
            case NodeType::ExportParquet:
                return ICON_IO_FILE;
            case NodeType::JSONFile:
            case NodeType::ARFFFile:
            case NodeType::ExportJSON:
                return ICON_IO_CODE;
            case NodeType::TXTFile:
            case NodeType::TextCorpusDataset:
                return ICON_IO_FILE;
            case NodeType::RESTAPISource:
                return ICON_IO_SERVER;
            case NodeType::ImageCSVDataset:
            case NodeType::ImageFolderDataset:
            case NodeType::FashionMNISTDataset:
            case NodeType::CIFAR100Dataset:
            case NodeType::ImagePreprocessor:
            case NodeType::QualityAnalyzer:
                return ICON_IO_IMAGE;

            // Data Transform Nodes
            case NodeType::SortRows:
                return ICON_IO_SORT;
            case NodeType::RemoveDuplicateRows:
                return ICON_IO_TRASH;
            case NodeType::RenameColumns:
            case NodeType::RowToColumnNames:
                return ICON_IO_EDIT;

            // ML Algorithms
            case NodeType::KMeansCluster:
            case NodeType::DBSCANCluster:
                return ICON_IO_CIRCLE;
            case NodeType::CrossValidationNode:
                return ICON_IO_COPY;
            case NodeType::OutlierDetector:
                return ICON_IO_BUG;

            case NodeType::PluginCustom:
                return ICON_IO_PLUG;

            default:
                return ICON_IO_NETWORK;
        }
    }

    // Phosphor Icon pack
    if (icon_pack_ == IconPack::Phosphor) {
        switch (type) {
            // Data Pipeline
            case NodeType::DatasetInput:
            case NodeType::DataLoader:
            case NodeType::ReplayBufferNode:
            case NodeType::SQLQuery:
            case NodeType::HDF5Dataset:
            case NodeType::StreamingDataset:
            case NodeType::TimeSeriesCSV:
            case NodeType::AudioFolderDataset:
            case NodeType::ExportSQL:
                return ICON_PH_DATABASE;
            case NodeType::Augmentation:
            case NodeType::DNNPreprocess:
            case NodeType::FillMissingValues:
                return ICON_PH_MAGIC_WAND;
            case NodeType::DataSplit:
            case NodeType::Split:
            case NodeType::TableSplitter:
            case NodeType::HierarchicalCluster:
            case NodeType::DecisionTreeClassifier:
                return ICON_PH_TREE_STRUCTURE;
            case NodeType::Normalize:
            case NodeType::BatchNorm:
            case NodeType::LayerNorm:
            case NodeType::GroupNorm:
            case NodeType::InstanceNorm:
            case NodeType::DescribeStats:
            case NodeType::FeatureImportanceNode:
                return ICON_PH_CHART_BAR;

            // Core Layers
            case NodeType::Dense:
            case NodeType::DNNModelLoad:
            case NodeType::PolicyNetwork:
            case NodeType::ValueNetwork:
            case NodeType::PretrainedYOLO:
            case NodeType::PretrainedMobileNet:
            case NodeType::PretrainedOpenPose:
            case NodeType::PretrainedFaceNet:
                return ICON_PH_BRAIN;
            case NodeType::Flatten:
            case NodeType::GroupByAggregate:
            case NodeType::UnionTables:
            case NodeType::RowAppender:
            case NodeType::RandomForestClassifier:
                return ICON_PH_STACK;

            // Convolutional Layers
            case NodeType::Conv1D:
            case NodeType::Conv2D:
            case NodeType::Conv3D:
            case NodeType::DepthwiseConv2D:
            case NodeType::PivotTable:
            case NodeType::Unpivot:
            case NodeType::CorrelationMatrix:
            case NodeType::SelectColumns:
            case NodeType::ColumnAppender:
            case NodeType::CrossTabulation:
            case NodeType::ConfusionMatrixNode:
            case NodeType::OneHotEncode:
                return ICON_PH_GRID_FOUR;

            // Pooling Layers
            case NodeType::MaxPool2D:
            case NodeType::AvgPool2D:
            case NodeType::GlobalMaxPool:
            case NodeType::GlobalAvgPool:
            case NodeType::AdaptiveAvgPool:
            case NodeType::Squeeze:
            case NodeType::Unsqueeze:
            case NodeType::TableCropper:
            case NodeType::PCANode:
            case NodeType::TSNENode:
            case NodeType::UMAPNode:
            case NodeType::Reshape:
            case NodeType::TensorReshape:
            case NodeType::View:
            case NodeType::ConvTranspose2D:
            case NodeType::Upsample:
            case NodeType::PixelShuffle:
                return ICON_PH_CUBE;

            // Regularization
            case NodeType::Dropout:
            case NodeType::SampleRows:
                return ICON_PH_SHUFFLE;

            // Recurrent Layers
            case NodeType::RNN:
            case NodeType::LSTM:
            case NodeType::GRU:
            case NodeType::Bidirectional:
            case NodeType::TimeDistributed:
                return ICON_PH_REPEAT;
            case NodeType::Embedding:
            case NodeType::TextTokenizer:
            case NodeType::StringManipulation:
                return ICON_PH_TEXT_T;
            case NodeType::TextVocabulary:
                return ICON_PH_LIST;
            case NodeType::TextPadding:
                return ICON_PH_TEXT_ALIGN_LEFT;

            // Attention & Transformer
            case NodeType::MultiHeadAttention:
            case NodeType::SelfAttention:
            case NodeType::CrossAttention:
            case NodeType::LinearAttention:
            case NodeType::GymEnvironment:
            case NodeType::KNNClassifier:
                return ICON_PH_TARGET;
            case NodeType::TransformerEncoder:
            case NodeType::TransformerDecoder:
                return ICON_PH_CPU;
            case NodeType::PositionalEncoding:
            case NodeType::ValueCounts:
                return ICON_PH_HASH;

            // Activation Functions
            case NodeType::ReLU:
            case NodeType::LeakyReLU:
            case NodeType::PReLU:
            case NodeType::ELU:
            case NodeType::SELU:
            case NodeType::GELU:
            case NodeType::Swish:
            case NodeType::Mish:
                return ICON_PH_LIGHTNING;
            case NodeType::Sigmoid:
            case NodeType::Spectrogram:
            case NodeType::MelSpectrogram:
            case NodeType::MFCC:
            case NodeType::SineWave:
            case NodeType::StepSignal:
            case NodeType::Tanh:
                return ICON_PH_WAVE_SINE;
            case NodeType::Softmax:
            case NodeType::GMMCluster:
            case NodeType::NaiveBayesClassifier:
                return ICON_PH_CHART_PIE;

            // Shape Operations
            case NodeType::Permute:
                return ICON_PH_SHUFFLE;

            // Merge Operations
            case NodeType::Concatenate:
            case NodeType::JoinTables:
                return ICON_PH_LINK;
            case NodeType::Add:
                return ICON_PH_PLUS;
            case NodeType::Multiply:
                return ICON_PH_X;
            case NodeType::Average:
            case NodeType::TimeSeriesWindow:
            case NodeType::TimeSeriesFeatures:
            case NodeType::TimeSeriesSplit:
            case NodeType::SignalScope:
            case NodeType::VisualizeData:
            case NodeType::LearningCurvesNode:
            case NodeType::ROCCurveNode:
            case NodeType::PRCurveNode:
            case NodeType::GradientBoostingClassifier:
            case NodeType::LogisticRegressionNode:
            case NodeType::LinearRegressionNode:
            case NodeType::PolynomialRegressionNode:
                return ICON_PH_CHART_LINE;

            // Output
            case NodeType::Output:
            case NodeType::DataValidator:
                return ICON_PH_CHECK_CIRCLE;

            // Loss Functions
            case NodeType::MSELoss:
            case NodeType::L1Loss:
            case NodeType::SmoothL1Loss:
            case NodeType::HuberLoss:
            case NodeType::MathFormula:
            case NodeType::RegressionMetricsNode:
                return ICON_PH_CALCULATOR;
            case NodeType::CrossEntropyLoss:
            case NodeType::BCELoss:
            case NodeType::BCEWithLogits:
            case NodeType::NLLLoss:
                return ICON_PH_SIGMA;

            // Optimizers & Scalers
            case NodeType::SGD:
            case NodeType::Adam:
            case NodeType::AdamW:
            case NodeType::RMSprop:
            case NodeType::Adagrad:
            case NodeType::NAdam:
            case NodeType::StandardScaler:
            case NodeType::MinMaxScaler:
            case NodeType::RobustScaler:
            case NodeType::SignalSlider:
            case NodeType::Parameter:
                return ICON_PH_GEAR;

            // Regularization Nodes
            case NodeType::L1Regularization:
            case NodeType::L2Regularization:
            case NodeType::ElasticNet:
            case NodeType::FilterRows:
            case NodeType::NonMaxSuppression:
            case NodeType::ThresholdFilter:
                return ICON_PH_FUNNEL;

            // Utility Nodes
            case NodeType::Lambda:
            case NodeType::RuleEngine:
                return ICON_PH_CODE;
            case NodeType::Identity:
                return ICON_PH_EQUALS;
            case NodeType::Constant:
            case NodeType::CellExtractor:
            case NodeType::CellUpdater:
            case NodeType::SVMClassifier:
            case NodeType::SVMRegressor:
                return ICON_PH_SQUARE;

            // Signal / Control
            case NodeType::RampSignal:
            case NodeType::ArgMax:
            case NodeType::TopK:
                return ICON_PH_TREND_UP;

            // Subgraph
            case NodeType::Subgraph:
                return ICON_PH_SQUARES;

            // DNN Inference Nodes
            case NodeType::DNNDetect:
            case NodeType::DNNFaceDetect:
            case NodeType::RLTraining:
                return ICON_PH_MAGNIFYING_GLASS;
            case NodeType::DNNClassify:
            case NodeType::LabelEncoder:
            case NodeType::OrdinalEncoder:
            case NodeType::TargetEncoder:
                return ICON_PH_TAG;
            case NodeType::DNNPoseEstimate:
                return ICON_PH_ROBOT;

            // Audio
            case NodeType::AudioInput:
            case NodeType::AudioAugmentation:
                return ICON_PH_MUSIC_NOTE;

            // Smart I/O Nodes
            case NodeType::DataInput:
                return ICON_PH_DOWNLOAD;
            case NodeType::DataOutput:
                return ICON_PH_EXPORT;

            // File Formats
            case NodeType::CSVFile:
            case NodeType::TSVFile:
            case NodeType::ExcelFile:
            case NodeType::ExportCSV:
            case NodeType::ExportExcel:
                return ICON_PH_FILE_XLS;
            case NodeType::ParquetFile:
            case NodeType::FeatherFile:
            case NodeType::ArrowIPCFile:
            case NodeType::NumPyFile:
            case NodeType::ExportParquet:
                return ICON_PH_FILE_TEXT;
            case NodeType::JSONFile:
            case NodeType::ARFFFile:
            case NodeType::ExportJSON:
                return ICON_PH_FILE_CODE;
            case NodeType::TXTFile:
            case NodeType::TextCorpusDataset:
                return ICON_PH_FILE_TEXT;
            case NodeType::RESTAPISource:
                return ICON_PH_SERVER;
            case NodeType::ImageCSVDataset:
            case NodeType::ImageFolderDataset:
            case NodeType::FashionMNISTDataset:
            case NodeType::CIFAR100Dataset:
            case NodeType::ImagePreprocessor:
            case NodeType::QualityAnalyzer:
                return ICON_PH_IMAGE;

            // Data Transform Nodes
            case NodeType::SortRows:
                return ICON_PH_SORT_ASCENDING;
            case NodeType::RemoveDuplicateRows:
                return ICON_PH_TRASH;
            case NodeType::RenameColumns:
            case NodeType::RowToColumnNames:
                return ICON_PH_PENCIL;

            // ML Algorithms
            case NodeType::KMeansCluster:
            case NodeType::DBSCANCluster:
                return ICON_PH_CIRCLE;
            case NodeType::CrossValidationNode:
                return ICON_PH_COPY;
            case NodeType::OutlierDetector:
                return ICON_PH_BUG;

            case NodeType::PluginCustom:
                return ICON_PH_PLUG;

            default:
                return ICON_PH_GRAPH;
        }
    }

    // Default: FontAwesome icon pack
    switch (type) {
        // Data Pipeline
        case NodeType::DatasetInput:
            return ICON_FA_DATABASE;
        case NodeType::DataLoader:
            return ICON_FA_SPINNER;
        case NodeType::Augmentation:
            return ICON_FA_WAND_MAGIC_SPARKLES;
        case NodeType::DataSplit:
            return ICON_FA_SITEMAP;
        case NodeType::Normalize:
            return ICON_FA_CHART_BAR;
        case NodeType::OneHotEncode:
            return ICON_FA_BORDER_ALL;

        // Core Layers
        case NodeType::Dense:
            return ICON_FA_BRAIN;
        case NodeType::Flatten:
            return ICON_FA_LAYER_GROUP;

        // Convolutional Layers
        case NodeType::Conv1D:
        case NodeType::Conv2D:
        case NodeType::Conv3D:
        case NodeType::DepthwiseConv2D:
            return ICON_FA_TABLE;

        // Pooling Layers
        case NodeType::MaxPool2D:
        case NodeType::AvgPool2D:
        case NodeType::GlobalMaxPool:
        case NodeType::GlobalAvgPool:
        case NodeType::AdaptiveAvgPool:
            return ICON_FA_COMPRESS;

        // Normalization Layers
        case NodeType::BatchNorm:
        case NodeType::LayerNorm:
        case NodeType::GroupNorm:
        case NodeType::InstanceNorm:
            return ICON_FA_CHART_BAR;

        // Regularization
        case NodeType::Dropout:
            return ICON_FA_DICE;

        // Recurrent Layers
        case NodeType::RNN:
        case NodeType::LSTM:
        case NodeType::GRU:
        case NodeType::Bidirectional:
        case NodeType::TimeDistributed:
            return ICON_FA_ROTATE;
        case NodeType::Embedding:
            return ICON_FA_FONT;

        // Attention & Transformer
        case NodeType::MultiHeadAttention:
        case NodeType::SelfAttention:
        case NodeType::CrossAttention:
        case NodeType::LinearAttention:
            return ICON_FA_BULLSEYE;
        case NodeType::TransformerEncoder:
        case NodeType::TransformerDecoder:
            return ICON_FA_MICROCHIP;
        case NodeType::PositionalEncoding:
            return ICON_FA_HASHTAG;

        // Activation Functions
        case NodeType::ReLU:
        case NodeType::LeakyReLU:
        case NodeType::PReLU:
        case NodeType::ELU:
        case NodeType::SELU:
        case NodeType::GELU:
        case NodeType::Swish:
        case NodeType::Mish:
            return ICON_FA_BOLT;
        case NodeType::Sigmoid:
            return ICON_FA_SIGNAL;
        case NodeType::Tanh:
            return ICON_FA_WAVE_SQUARE;
        case NodeType::Softmax:
            return ICON_FA_CHART_PIE;

        // Shape Operations
        case NodeType::Reshape:
        case NodeType::TensorReshape:
        case NodeType::View:
            return ICON_FA_EXPAND;
        case NodeType::Permute:
            return ICON_FA_SHUFFLE;
        case NodeType::Squeeze:
        case NodeType::Unsqueeze:
            return ICON_FA_COMPRESS;
        case NodeType::Split:
            return ICON_FA_SITEMAP;

        // Merge Operations
        case NodeType::Concatenate:
            return ICON_FA_LINK;
        case NodeType::Add:
            return ICON_FA_PLUS;
        case NodeType::Multiply:
            return ICON_FA_XMARK;
        case NodeType::Average:
            return ICON_FA_CHART_LINE;

        // Output
        case NodeType::Output:
            return ICON_FA_CIRCLE_CHECK;

        // Loss Functions
        case NodeType::MSELoss:
        case NodeType::L1Loss:
        case NodeType::SmoothL1Loss:
        case NodeType::HuberLoss:
            return ICON_FA_CALCULATOR;
        case NodeType::CrossEntropyLoss:
        case NodeType::BCELoss:
        case NodeType::BCEWithLogits:
        case NodeType::NLLLoss:
            return ICON_FA_SCALE_BALANCED;

        // Optimizers
        case NodeType::SGD:
        case NodeType::Adam:
        case NodeType::AdamW:
        case NodeType::RMSprop:
        case NodeType::Adagrad:
        case NodeType::NAdam:
            return ICON_FA_SLIDERS;

        // Learning Rate Schedulers
        case NodeType::StepLR:
        case NodeType::CosineAnnealing:
        case NodeType::ReduceOnPlateau:
        case NodeType::ExponentialLR:
        case NodeType::WarmupScheduler:
            return ICON_FA_CHART_LINE;

        // Regularization Nodes
        case NodeType::L1Regularization:
        case NodeType::L2Regularization:
        case NodeType::ElasticNet:
            return ICON_FA_SHIELD;

        // Utility Nodes
        case NodeType::Lambda:
            return ICON_FA_CODE;
        case NodeType::Identity:
            return ICON_FA_EQUALS;
        case NodeType::Constant:
            return ICON_FA_CIRCLE;
        case NodeType::Parameter:
            return ICON_FA_GEAR;

        // Signal / Control
        case NodeType::SignalSlider:
            return ICON_FA_SLIDERS;
        case NodeType::SineWave:
            return ICON_FA_WAVE_SQUARE;
        case NodeType::StepSignal:
            return ICON_FA_WAVE_SQUARE;
        case NodeType::RampSignal:
            return ICON_FA_ARROW_TREND_UP;
        case NodeType::SignalScope:
            return ICON_FA_CHART_LINE;

        // Subgraph
        case NodeType::Subgraph:
            return ICON_FA_OBJECT_GROUP;

        // DNN Inference Nodes
        case NodeType::DNNModelLoad:
            return ICON_FA_BRAIN;
        case NodeType::DNNDetect:
        case NodeType::DNNFaceDetect:
            return ICON_FA_CROSSHAIRS;
        case NodeType::DNNClassify:
            return ICON_FA_TAGS;
        case NodeType::DNNPoseEstimate:
            return ICON_FA_USER;
        case NodeType::DNNPreprocess:
            return ICON_FA_WAND_MAGIC_SPARKLES;
        case NodeType::PretrainedYOLO:
        case NodeType::PretrainedMobileNet:
        case NodeType::PretrainedOpenPose:
        case NodeType::PretrainedFaceNet:
            return ICON_FA_BRAIN;
        case NodeType::NonMaxSuppression:
        case NodeType::ThresholdFilter:
            return ICON_FA_FILTER;
        case NodeType::ArgMax:
        case NodeType::TopK:
            return ICON_FA_ARROW_UP;

        // Text Processing
        case NodeType::TextTokenizer:
            return ICON_FA_FONT;
        case NodeType::TextVocabulary:
            return ICON_FA_BOOK;
        case NodeType::TextPadding:
            return ICON_FA_ALIGN_LEFT;
        case NodeType::NERSequenceBuilder:
        case NodeType::NERTagVocabulary:
        case NodeType::SequenceTagOutput:
            return ICON_FA_TAG;
        case NodeType::TokenVocabulary:
        case NodeType::POSVocabulary:
            return ICON_FA_BOOK;
        case NodeType::PairDatasetBuilder:
        case NodeType::TripletDatasetBuilder:
        case NodeType::SiameseBranch:
            return ICON_FA_CODE_BRANCH;
        case NodeType::SharedEncoder:
            return ICON_FA_SHARE_NODES;
        case NodeType::ContrastiveLoss:
        case NodeType::CosineEmbeddingLoss:
        case NodeType::TripletLoss:
            return ICON_FA_SCALE_BALANCED;
        case NodeType::PairMetrics:
        case NodeType::RetrievalMetrics:
        case NodeType::PairScoreOutput:
            return ICON_FA_CHART_LINE;
        case NodeType::EmbeddingOutput:
            return ICON_FA_CUBE;

        // Upsampling
        case NodeType::ConvTranspose2D:
        case NodeType::Upsample:
        case NodeType::PixelShuffle:
            return ICON_FA_EXPAND;

        // Time-Series
        case NodeType::TimeSeriesWindow:
        case NodeType::TimeSeriesFeatures:
        case NodeType::TimeSeriesSplit:
            return ICON_FA_CHART_LINE;

        // Audio
        case NodeType::AudioInput:
        case NodeType::AudioAugmentation:
            return ICON_FA_MUSIC;
        case NodeType::Spectrogram:
        case NodeType::MelSpectrogram:
        case NodeType::MFCC:
            return ICON_FA_WAVE_SQUARE;

        // RL
        case NodeType::GymEnvironment:
            return ICON_FA_BULLSEYE;
        case NodeType::ReplayBufferNode:
            return ICON_FA_DATABASE;
        case NodeType::PolicyNetwork:
        case NodeType::ValueNetwork:
            return ICON_FA_BRAIN;
        case NodeType::RLTraining:
            return ICON_FA_CROSSHAIRS;

        // Smart I/O Nodes
        case NodeType::DataInput:
            return ICON_FA_FILE_IMPORT;
        case NodeType::DataOutput:
            return ICON_FA_FILE_EXPORT;
        case NodeType::DataConvert:
            return ICON_FA_RIGHT_LEFT;

        // Legacy Data Source Nodes (File Formats)
        case NodeType::CSVFile:
        case NodeType::TSVFile:
        case NodeType::ExcelFile:
            return ICON_FA_FILE_EXCEL;
        case NodeType::SQLQuery:
        case NodeType::HDF5Dataset:
            return ICON_FA_DATABASE;
        case NodeType::ParquetFile:
        case NodeType::FeatherFile:
        case NodeType::ArrowIPCFile:
        case NodeType::NumPyFile:
            return ICON_FA_FILE_LINES;
        case NodeType::JSONFile:
        case NodeType::ARFFFile:
            return ICON_FA_FILE_CODE;
        case NodeType::TXTFile:
        case NodeType::TextCorpusDataset:
            return ICON_FA_FILE_LINES;
        case NodeType::RESTAPISource:
            return ICON_FA_GLOBE;
        case NodeType::ImageCSVDataset:
        case NodeType::ImageFolderDataset:
        case NodeType::FashionMNISTDataset:
        case NodeType::CIFAR100Dataset:
            return ICON_FA_IMAGES;
        case NodeType::StreamingDataset:
        case NodeType::TimeSeriesCSV:
        case NodeType::AudioFolderDataset:
            return ICON_FA_DATABASE;

        // Data Transform Nodes
        case NodeType::FilterRows:
            return ICON_FA_FILTER;
        case NodeType::SelectColumns:
            return ICON_FA_TABLE_COLUMNS;
        case NodeType::JoinTables:
            return ICON_FA_CODE_BRANCH;
        case NodeType::GroupByAggregate:
            return ICON_FA_LAYER_GROUP;
        case NodeType::SortRows:
            return ICON_FA_SORT;
        case NodeType::FillMissingValues:
            return ICON_FA_WAND_MAGIC_SPARKLES;
        case NodeType::RemoveDuplicateRows:
            return ICON_FA_TRASH;
        case NodeType::PivotTable:
        case NodeType::Unpivot:
            return ICON_FA_TABLE;
        case NodeType::UnionTables:
        case NodeType::RowAppender:
            return ICON_FA_LAYER_GROUP;
        case NodeType::RenameColumns:
            return ICON_FA_PEN;

        // Analytics Nodes
        case NodeType::DescribeStats:
            return ICON_FA_CHART_BAR;
        case NodeType::VisualizeData:
            return ICON_FA_CHART_LINE;
        case NodeType::SampleRows:
            return ICON_FA_DICE;
        case NodeType::CorrelationMatrix:
            return ICON_FA_TABLE;
        case NodeType::ValueCounts:
            return ICON_FA_HASHTAG;
        case NodeType::CrossTabulation:
            return ICON_FA_TH;

        // Data Export Nodes
        case NodeType::ExportCSV:
        case NodeType::ExportExcel:
            return ICON_FA_FILE_EXCEL;
        case NodeType::ExportParquet:
            return ICON_FA_FILE_LINES;
        case NodeType::ExportSQL:
            return ICON_FA_DATABASE;
        case NodeType::ExportJSON:
            return ICON_FA_FILE_CODE;

        // KNIME-Style Table Manipulation Nodes
        case NodeType::RowToColumnNames:
            return ICON_FA_BARS;
        case NodeType::TableSplitter:
            return ICON_FA_SITEMAP;
        case NodeType::CellExtractor:
        case NodeType::CellUpdater:
            return ICON_FA_SQUARE;
        case NodeType::TableCropper:
            return ICON_FA_COMPRESS;
        case NodeType::ColumnAppender:
            return ICON_FA_TABLE_COLUMNS;
        case NodeType::StringManipulation:
            return ICON_FA_FONT;
        case NodeType::MathFormula:
            return ICON_FA_CALCULATOR;
        case NodeType::RuleEngine:
            return ICON_FA_CODE;

        // Machine Learning Algorithms - Clustering
        case NodeType::KMeansCluster:
            return ICON_FA_CIRCLE_NODES;
        case NodeType::DBSCANCluster:
            return ICON_FA_CIRCLE_NODES;
        case NodeType::HierarchicalCluster:
            return ICON_FA_SITEMAP;
        case NodeType::GMMCluster:
            return ICON_FA_CHART_PIE;

        // Machine Learning - Dimensionality Reduction
        case NodeType::PCANode:
        case NodeType::TSNENode:
        case NodeType::UMAPNode:
            return ICON_FA_COMPRESS;

        // Machine Learning - Classification
        case NodeType::DecisionTreeClassifier:
            return ICON_FA_SITEMAP;
        case NodeType::RandomForestClassifier:
            return ICON_FA_CUBES;
        case NodeType::GradientBoostingClassifier:
            return ICON_FA_CHART_LINE;
        case NodeType::TreeModelPredictor:
            return ICON_FA_FILE_IMPORT;
        case NodeType::SVMClassifier:
        case NodeType::SVMRegressor:
            return ICON_FA_SQUARE;
        case NodeType::KNNClassifier:
            return ICON_FA_BULLSEYE;
        case NodeType::NaiveBayesClassifier:
            return ICON_FA_CHART_PIE;
        case NodeType::LogisticRegressionNode:
        case NodeType::LinearRegressionNode:
        case NodeType::PolynomialRegressionNode:
            return ICON_FA_CHART_LINE;

        // Model Evaluation Nodes
        case NodeType::ConfusionMatrixNode:
            return ICON_FA_TH;
        case NodeType::ROCCurveNode:
        case NodeType::PRCurveNode:
            return ICON_FA_CHART_LINE;
        case NodeType::LearningCurvesNode:
            return ICON_FA_CHART_LINE;
        case NodeType::FeatureImportanceNode:
            return ICON_FA_CHART_BAR;
        case NodeType::CrossValidationNode:
            return ICON_FA_COPY;
        case NodeType::RegressionMetricsNode:
            return ICON_FA_CALCULATOR;

        // Data Preprocessing Nodes
        case NodeType::StandardScaler:
        case NodeType::MinMaxScaler:
        case NodeType::RobustScaler:
            return ICON_FA_SLIDERS;
        case NodeType::LabelEncoder:
        case NodeType::OrdinalEncoder:
        case NodeType::TargetEncoder:
            return ICON_FA_TAGS;

        // Advanced Preprocessing
        case NodeType::OutlierDetector:
            return ICON_FA_BUG;
        case NodeType::ImagePreprocessor:
        case NodeType::QualityAnalyzer:
            return ICON_FA_IMAGES;
        case NodeType::DataValidator:
            return ICON_FA_CHECK;

        case NodeType::PluginCustom:
            return ICON_FA_PLUG;

        default:
            return ICON_FA_CIRCLE_NODES;
    }
}

} // namespace gui
