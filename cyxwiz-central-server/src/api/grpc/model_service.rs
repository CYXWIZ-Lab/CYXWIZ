use crate::database::{
    models::{Model, ModelFormat, ModelSource},
    queries, DbPool,
};
use crate::error::ServerError;
use chrono::Utc;
use sha2::{Digest, Sha256};
use std::path::PathBuf;
use std::pin::Pin;
use tokio::fs::{self, File};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio_stream::{wrappers::ReceiverStream, Stream, StreamExt};
use tonic::{Request, Response, Status};
use tracing::{error, info, warn};
use uuid::Uuid;

// Import generated proto types
pub mod pb {
    tonic::include_proto!("cyxwiz.protocol");
}

use pb::{
    model_service_server::ModelService, DeleteModelRequest, DeleteModelResponse,
    DownloadModelMetadataRequest, DownloadModelMetadataResponse, ListModelsRequest,
    ListModelsResponse, ModelChunk, ModelFormat as PbModelFormat, ModelInfo as PbModelInfo,
    ModelRegistryEntry as PbModelRegistryEntry, ModelSource as PbModelSource, SimpleResponse,
    StatusCode, UploadModelMetadataRequest, UploadModelMetadataResponse,
};

type ModelChunkStream = Pin<Box<dyn Stream<Item = Result<ModelChunk, Status>> + Send>>;

const CHUNK_SIZE: usize = 1024 * 1024; // 1 MB chunks
const MAX_MODEL_SIZE: u64 = 10 * 1024 * 1024 * 1024; // 10 GB max

pub struct ModelServiceImpl {
    db_pool: DbPool,
    storage_path: PathBuf,
}

impl ModelServiceImpl {
    pub fn new(db_pool: DbPool, storage_path: PathBuf) -> Self {
        Self {
            db_pool,
            storage_path,
        }
    }

    /// Convert protobuf model format to database enum
    fn pb_format_to_db(format: PbModelFormat) -> Result<ModelFormat, Status> {
        Ok(match format {
            PbModelFormat::ModelFormatOnnx => ModelFormat::Onnx,
            PbModelFormat::ModelFormatGguf => ModelFormat::Gguf,
            PbModelFormat::ModelFormatPytorch => ModelFormat::Pytorch,
            PbModelFormat::ModelFormatTensorflow => ModelFormat::Tensorflow,
            PbModelFormat::ModelFormatSafetensors => ModelFormat::Safetensors,
            PbModelFormat::ModelFormatTflite => ModelFormat::Tflite,
            PbModelFormat::ModelFormatTorchscript => ModelFormat::Torchscript,
            _ => return Err(Status::invalid_argument("Invalid model format")),
        })
    }

    /// Convert protobuf model source to database enum
    fn pb_source_to_db(source: PbModelSource) -> Result<ModelSource, Status> {
        Ok(match source {
            PbModelSource::ModelSourceLocal => ModelSource::Local,
            PbModelSource::ModelSourceHuggingface => ModelSource::Huggingface,
            PbModelSource::ModelSourceCyxwizHub => ModelSource::CyxwizHub,
            PbModelSource::ModelSourceUrl => ModelSource::Url,
            _ => return Err(Status::invalid_argument("Invalid model source")),
        })
    }

    /// Convert database model format to protobuf
    fn db_format_to_pb(format: &ModelFormat) -> i32 {
        match format {
            ModelFormat::Onnx => PbModelFormat::ModelFormatOnnx as i32,
            ModelFormat::Gguf => PbModelFormat::ModelFormatGguf as i32,
            ModelFormat::Pytorch => PbModelFormat::ModelFormatPytorch as i32,
            ModelFormat::Tensorflow => PbModelFormat::ModelFormatTensorflow as i32,
            ModelFormat::Safetensors => PbModelFormat::ModelFormatSafetensors as i32,
            ModelFormat::Tflite => PbModelFormat::ModelFormatTflite as i32,
            ModelFormat::Torchscript => PbModelFormat::ModelFormatTorchscript as i32,
        }
    }

    /// Convert database model source to protobuf
    fn db_source_to_pb(source: &ModelSource) -> i32 {
        match source {
            ModelSource::Local => PbModelSource::ModelSourceLocal as i32,
            ModelSource::Huggingface => PbModelSource::ModelSourceHuggingface as i32,
            ModelSource::CyxwizHub => PbModelSource::ModelSourceCyxwizHub as i32,
            ModelSource::Url => PbModelSource::ModelSourceUrl as i32,
        }
    }

    /// Convert database model to protobuf registry entry
    fn model_to_pb(&self, model: &Model) -> PbModelRegistryEntry {
        PbModelRegistryEntry {
            model_id: model.id.to_string(),
            info: Some(PbModelInfo {
                model_id: model.id.to_string(),
                name: model.name.clone(),
                description: model.description.clone().unwrap_or_default(),
                format: Self::db_format_to_pb(&model.format),
                source: Self::db_source_to_pb(&model.source),
                source_url: model.source_url.clone().unwrap_or_default(),
                local_path: String::new(), // Don't expose internal storage path
                size_bytes: model.size_bytes,
                requirements: None, // TODO: Add hardware requirements
                metadata: std::collections::HashMap::new(),
            }),
            owner_id: model.owner_user_id.clone(),
            is_public: model.is_public,
            download_count: model.download_count,
            rating: model.rating,
            tags: model.tags.clone(),
            uploaded_at: model.created_at.timestamp(),
            storage_path: String::new(), // Don't expose internal path
        }
    }

    /// Get model storage path
    fn get_model_path(&self, model_id: Uuid) -> PathBuf {
        self.storage_path.join(model_id.to_string())
    }
}

#[tonic::async_trait]
impl ModelService for ModelServiceImpl {
    async fn upload_metadata(
        &self,
        request: Request<UploadModelMetadataRequest>,
    ) -> Result<Response<UploadModelMetadataResponse>, Status> {
        let req = request.into_inner();

        let model_info = req
            .info
            .ok_or_else(|| Status::invalid_argument("Model info is required"))?;

        info!("Uploading model metadata: {}", model_info.name);

        // Parse and validate
        let format = Self::pb_format_to_db(PbModelFormat::try_from(model_info.format).unwrap_or(PbModelFormat::ModelFormatUnknown))?;
        let source = Self::pb_source_to_db(PbModelSource::try_from(model_info.source).unwrap_or(PbModelSource::ModelSourceUnknown))?;

        // Validate size
        if model_info.size_bytes > MAX_MODEL_SIZE as i64 {
            return Err(Status::invalid_argument(format!(
                "Model size exceeds maximum allowed ({}GB)",
                MAX_MODEL_SIZE / (1024 * 1024 * 1024)
            )));
        }

        // Create model record
        let model_id = Uuid::new_v4();
        let storage_path = self.get_model_path(model_id);

        let model = Model {
            id: model_id,
            name: model_info.name.clone(),
            description: if model_info.description.is_empty() {
                None
            } else {
                Some(model_info.description.clone())
            },
            owner_user_id: req.user_id.clone(),
            format,
            source,
            source_url: if model_info.source_url.is_empty() {
                None
            } else {
                Some(model_info.source_url.clone())
            },
            size_bytes: model_info.size_bytes,
            min_vram_bytes: 0, // TODO: Extract from requirements
            min_ram_bytes: 0,
            min_cpu_cores: 1,
            required_device_type: None,
            gpu_preference: None,
            is_public: req.is_public,
            price_per_download: 0, // TODO: Add pricing
            download_count: 0,
            rating: 0.0,
            rating_count: 0,
            tags: req.tags.clone(),
            storage_path: storage_path.to_string_lossy().to_string(),
            checksum_sha256: String::new(), // Will be set after upload
            metadata: serde_json::json!(model_info.metadata),
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };

        match queries::create_model(&self.db_pool, &model).await {
            Ok(_) => {
                info!("Model {} metadata created successfully", model_id);

                Ok(Response::new(UploadModelMetadataResponse {
                    status: StatusCode::StatusSuccess as i32,
                    model_id: model_id.to_string(),
                    upload_url: format!("/upload/{}", model_id), // Placeholder URL
                    error: None,
                }))
            }
            Err(e) => {
                error!("Failed to create model metadata: {}", e);
                Err(Status::internal(format!("Database error: {}", e)))
            }
        }
    }

    async fn upload_model(
        &self,
        request: Request<tonic::Streaming<ModelChunk>>,
    ) -> Result<Response<SimpleResponse>, Status> {
        let mut stream = request.into_inner();

        let mut model_id: Option<Uuid> = None;
        let mut file: Option<File> = None;
        let mut hasher = Sha256::new();
        let mut total_bytes: i64 = 0;

        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result?;

            // First chunk should contain model_id
            if model_id.is_none() {
                model_id = Some(
                    Uuid::parse_str(&chunk.model_id)
                        .map_err(|e| Status::invalid_argument(format!("Invalid model ID: {}", e)))?,
                );

                info!("Starting model upload: {}", model_id.as_ref().unwrap());

                // Create storage directory if it doesn't exist
                let model_path = self.get_model_path(model_id.unwrap());
                if let Some(parent) = model_path.parent() {
                    fs::create_dir_all(parent)
                        .await
                        .map_err(|e| Status::internal(format!("Failed to create storage directory: {}", e)))?;
                }

                // Create file for writing
                file = Some(
                    File::create(&model_path)
                        .await
                        .map_err(|e| Status::internal(format!("Failed to create file: {}", e)))?,
                );
            }

            // Write chunk to file
            if let Some(ref mut f) = file {
                f.write_all(&chunk.data)
                    .await
                    .map_err(|e| Status::internal(format!("Failed to write chunk: {}", e)))?;

                // Update hash
                hasher.update(&chunk.data);
                total_bytes += chunk.data.len() as i64;
            }

            // Check if this is the final chunk
            if chunk.is_final {
                info!("Model upload completed: {} bytes", total_bytes);

                // Finalize hash
                let checksum = format!("{:x}", hasher.finalize());

                // Verify checksum if provided
                if !chunk.checksum.is_empty() && chunk.checksum != checksum {
                    error!("Checksum mismatch: expected {}, got {}", chunk.checksum, checksum);

                    // Delete the file
                    if let Some(mid) = model_id {
                        let _ = fs::remove_file(self.get_model_path(mid)).await;
                    }

                    return Err(Status::data_loss("Checksum verification failed"));
                }

                // TODO: Update model record with checksum and actual size

                break;
            }
        }

        if model_id.is_none() {
            return Err(Status::invalid_argument("No data received"));
        }

        info!("Model upload successful: {}", model_id.unwrap());

        Ok(Response::new(SimpleResponse {
            status: StatusCode::StatusSuccess as i32,
            error: None,
        }))
    }

    async fn download_metadata(
        &self,
        request: Request<DownloadModelMetadataRequest>,
    ) -> Result<Response<DownloadModelMetadataResponse>, Status> {
        let req = request.into_inner();

        let model_id = Uuid::parse_str(&req.model_id)
            .map_err(|e| Status::invalid_argument(format!("Invalid model ID: {}", e)))?;

        match queries::get_model_by_id(&self.db_pool, model_id).await {
            Ok(model) => {
                let pb_entry = self.model_to_pb(&model);

                Ok(Response::new(DownloadModelMetadataResponse {
                    status: StatusCode::StatusSuccess as i32,
                    entry: Some(pb_entry),
                    download_url: format!("/download/{}", model_id), // Placeholder URL
                    error: None,
                }))
            }
            Err(ServerError::NotFound(msg)) => Err(Status::not_found(msg)),
            Err(e) => Err(Status::internal(format!("Database error: {}", e))),
        }
    }

    type DownloadModelStream = ModelChunkStream;

    async fn download_model(
        &self,
        request: Request<DownloadModelMetadataRequest>,
    ) -> Result<Response<Self::DownloadModelStream>, Status> {
        let req = request.into_inner();

        let model_id = Uuid::parse_str(&req.model_id)
            .map_err(|e| Status::invalid_argument(format!("Invalid model ID: {}", e)))?;

        // Get model metadata
        let model = match queries::get_model_by_id(&self.db_pool, model_id).await {
            Ok(m) => m,
            Err(ServerError::NotFound(msg)) => return Err(Status::not_found(msg)),
            Err(e) => return Err(Status::internal(format!("Database error: {}", e))),
        };

        info!("Starting model download: {} by user", model_id);

        // Check file exists
        let model_path = self.get_model_path(model_id);
        if !model_path.exists() {
            return Err(Status::not_found("Model file not found in storage"));
        }

        // Open file for reading
        let mut file = File::open(&model_path)
            .await
            .map_err(|e| Status::internal(format!("Failed to open model file: {}", e)))?;

        // Get file size
        let total_size = model.size_bytes;

        // Create channel for streaming
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<ModelChunk, Status>>(10);

        // Spawn task to read and stream file
        tokio::spawn(async move {
            let mut offset: i64 = 0;
            let mut buffer = vec![0u8; CHUNK_SIZE];
            let mut hasher = Sha256::new();

            loop {
                match file.read(&mut buffer).await {
                    Ok(0) => {
                        // EOF reached, send final chunk
                        let checksum = format!("{:x}", hasher.finalize());

                        let final_chunk = ModelChunk {
                            model_id: model_id.to_string(),
                            data: vec![],
                            offset,
                            total_size,
                            is_final: true,
                            checksum,
                        };

                        if tx.send(Ok(final_chunk)).await.is_err() {
                            error!("Failed to send final chunk");
                        }
                        break;
                    }
                    Ok(n) => {
                        let data = buffer[..n].to_vec();
                        hasher.update(&data);

                        let chunk = ModelChunk {
                            model_id: model_id.to_string(),
                            data,
                            offset,
                            total_size,
                            is_final: false,
                            checksum: String::new(),
                        };

                        if tx.send(Ok(chunk)).await.is_err() {
                            error!("Failed to send chunk");
                            break;
                        }

                        offset += n as i64;
                    }
                    Err(e) => {
                        error!("Failed to read file: {}", e);
                        let _ = tx.send(Err(Status::internal("Failed to read file"))).await;
                        break;
                    }
                }
            }
        });

        // TODO: Record download in database

        let out_stream = ReceiverStream::new(rx);
        Ok(Response::new(Box::pin(out_stream) as Self::DownloadModelStream))
    }

    async fn list_models(
        &self,
        request: Request<ListModelsRequest>,
    ) -> Result<Response<ListModelsResponse>, Status> {
        let req = request.into_inner();

        let page_size = if req.page_size > 0 {
            req.page_size.min(100) as i64
        } else {
            20
        };

        // Parse page token as offset
        let offset: i64 = req.page_token.parse().unwrap_or(0);

        let search_query = if req.search_query.is_empty() {
            None
        } else {
            Some(req.search_query.as_str())
        };

        match queries::list_models(
            &self.db_pool,
            search_query,
            &req.tags,
            req.public_only,
            page_size,
            offset,
        )
        .await
        {
            Ok(models) => {
                let pb_models: Vec<PbModelRegistryEntry> =
                    models.iter().map(|m| self.model_to_pb(m)).collect();

                let next_page_token = if models.len() == page_size as usize {
                    (offset + page_size).to_string()
                } else {
                    String::new()
                };

                Ok(Response::new(ListModelsResponse {
                    models: pb_models,
                    next_page_token,
                    total_count: 0, // TODO: Get actual count
                    error: None,
                }))
            }
            Err(e) => Err(Status::internal(format!("Database error: {}", e))),
        }
    }

    async fn delete_model(
        &self,
        request: Request<DeleteModelRequest>,
    ) -> Result<Response<DeleteModelResponse>, Status> {
        let req = request.into_inner();

        let model_id = Uuid::parse_str(&req.model_id)
            .map_err(|e| Status::invalid_argument(format!("Invalid model ID: {}", e)))?;

        info!("Deleting model {} by user {}", model_id, req.user_id);

        // Delete from database (checks ownership)
        match queries::delete_model(&self.db_pool, model_id, &req.user_id).await {
            Ok(_) => {
                // Delete file from storage
                let model_path = self.get_model_path(model_id);
                if model_path.exists() {
                    if let Err(e) = fs::remove_file(&model_path).await {
                        warn!("Failed to delete model file: {}", e);
                    }
                }

                info!("Model {} deleted successfully", model_id);

                Ok(Response::new(DeleteModelResponse {
                    status: StatusCode::StatusSuccess as i32,
                    error: None,
                }))
            }
            Err(ServerError::NotFound(msg)) => Err(Status::not_found(msg)),
            Err(e) => {
                error!("Failed to delete model: {}", e);
                Err(Status::internal(format!("Database error: {}", e)))
            }
        }
    }
}
