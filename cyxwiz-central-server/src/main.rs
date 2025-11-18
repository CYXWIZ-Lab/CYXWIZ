mod api;
mod blockchain;
mod cache;
mod config;
mod database;
mod error;
mod pb; // Needed by scheduler for gRPC client
mod scheduler;
mod tui;

use crate::api::grpc::{JobServiceImpl, JobStatusServiceImpl, NodeServiceImpl};
use crate::blockchain::{PaymentProcessor, SolanaClient};
use crate::cache::RedisCache;
use crate::config::Config;
use crate::database::{create_pool, run_migrations};
use crate::scheduler::{JobScheduler, NodeMonitor};
use std::sync::Arc;
use tokio::sync::RwLock;
use tonic::transport::Server;
use tracing::{error, info, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Running in TUI-only mode until gRPC compilation issues are resolved
    // See GRPC_ENABLEMENT_GUIDE.md for details

    // Initialize tracing
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "cyxwiz_central_server=info,tower_http=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    info!("CyxWiz Central Server v{}", env!("CARGO_PKG_VERSION"));
    info!("========================================");

    // Load configuration
    info!("Loading configuration...");
    let config = Config::from_file("config.toml").unwrap_or_else(|e| {
        error!("Failed to load config: {}, using defaults", e);
        Config::default()
    });

    // Initialize database
    info!("Connecting to database: {}", config.database.url);
    let db_pool = create_pool(&config.database).await?;

    // Run migrations
    info!("Running database migrations...");
    if let Err(e) = run_migrations(&db_pool).await {
        error!("Failed to run migrations: {}", e);
        error!("Please ensure PostgreSQL is running and the database exists");
        return Err(e.into());
    }
    info!("Migrations completed");

    // Initialize Redis cache (optional for SQLite quick test)
    info!("Attempting to connect to Redis: {}", config.redis.url);
    let redis_cache_result = RedisCache::new(&config.redis).await;

    let redis_cache = match redis_cache_result {
        Ok(cache) => {
            info!("✓ Redis connected successfully");
            cache
        }
        Err(e) => {
            warn!("⚠ Redis connection failed: {}", e);
            warn!("  Running in MOCK MODE - Redis features disabled");
            warn!("  TUI will function but caching operations will fail gracefully");
            info!("");
            info!("  To enable full Redis functionality:");
            info!("  1. Install Redis:");
            info!("     - Windows: Download from https://github.com/microsoftarchive/redis/releases");
            info!("     - WSL: sudo apt-get install redis-server && redis-server");
            info!("  2. Use Docker: docker run -d -p 6379:6379 redis:alpine");
            info!("");
            // Create mock cache that will work without Redis
            RedisCache::new_mock()
        }
    };
    let redis_cache_arc = Arc::new(RwLock::new(redis_cache));

    // Initialize job scheduler (runs in both TUI and gRPC modes)
    info!("Starting job scheduler...");
    let scheduler = Arc::new(JobScheduler::new(
        db_pool.clone(),
        redis_cache_arc.read().await.clone(),
        config.scheduler.clone(),
    ));

    // Start scheduler loop in background
    let scheduler_clone = Arc::clone(&scheduler);
    tokio::spawn(async move {
        scheduler_clone.run().await;
    });
    info!("Job scheduler started");

    // Start node monitor in background to detect disconnected nodes
    info!("Starting node monitor...");
    let node_monitor = NodeMonitor::new(
        db_pool.clone(),
        30,  // 30 seconds timeout - mark offline if no heartbeat
        10   // Check every 10 seconds
    );
    tokio::spawn(async move {
        node_monitor.run().await;
    });
    info!("Node monitor started");

    // Check command line arguments for mode
    let args: Vec<String> = std::env::args().collect();
    let tui_mode = args.iter().any(|arg| arg == "--tui" || arg == "-t");

    if tui_mode {
        // TUI mode with scheduler
        info!("Starting in TUI mode with live job processing...");
        info!("========================================");
        return tui::run(db_pool, redis_cache_arc).await.map_err(|e| e.into());
    }

    // gRPC and REST server mode (default)
    // Initialize Solana blockchain client
    info!("Initializing Solana client...");
    let solana_client = if std::path::Path::new(&config.blockchain.payer_keypair_path).exists() {
        match SolanaClient::from_keypair_file(
            &config.blockchain.solana_rpc_url,
            &config.blockchain.payer_keypair_path,
            &config.blockchain.program_id,
        ) {
            Ok(client) => {
                info!("Solana client initialized (network: {})", config.blockchain.network);
                info!("Payer pubkey: {}", client.payer_pubkey());
                Some(client)
            }
            Err(e) => {
                error!("Failed to initialize Solana client: {}", e);
                error!("Payment processing will be disabled");
                None
            }
        }
    } else {
        error!("Solana keypair file not found: {}", config.blockchain.payer_keypair_path);
        error!("Payment processing will be disabled");
        None
    };

    // Initialize payment processor
    let payment_processor = if let Some(client) = solana_client {
        Arc::new(PaymentProcessor::new(client))
    } else {
        // Use a dummy client for development (mocked Keypair is Vec<u8>)
        let dummy_keypair = Vec::new();
        let dummy_client = SolanaClient::new(
            "https://api.devnet.solana.com",
            dummy_keypair,
            "11111111111111111111111111111111",
        )?;
        Arc::new(PaymentProcessor::new(dummy_client))
    };

    // Initialize gRPC services
    let job_service = JobServiceImpl::new(
        db_pool.clone(),
        Arc::clone(&scheduler),
        Arc::clone(&payment_processor),
    );
    let node_service = NodeServiceImpl::new(db_pool.clone(), Arc::clone(&scheduler));
    let job_status_service = JobStatusServiceImpl::new(db_pool.clone());

    // Initialize deployment services
    // let deployment_service = DeploymentServiceImpl::new(db_pool.clone()); // Temporarily disabled
    // let terminal_service = TerminalServiceImpl::new(db_pool.clone()); // Temporarily disabled
    // let model_service = ModelServiceImpl::new(  // Temporarily disabled
    //     db_pool.clone(),
    //     std::path::PathBuf::from("./storage/models"),
    // );

    // Build gRPC server
    let grpc_addr = config.server.grpc_address.parse()?;
    info!("Starting gRPC server on {}", grpc_addr);

    let grpc_server = Server::builder()
        .add_service(pb::job_service_server::JobServiceServer::new(job_service))
        .add_service(pb::node_service_server::NodeServiceServer::new(node_service))
        .add_service(pb::job_status_service_server::JobStatusServiceServer::new(job_status_service))
        // .add_service(pb::deployment_service_server::DeploymentServiceServer::new(deployment_service)) // Temporarily disabled
        // .add_service(pb::terminal_service_server::TerminalServiceServer::new(terminal_service)) // Temporarily disabled
        // .add_service(pb::model_service_server::ModelServiceServer::new(model_service)) // Temporarily disabled
        .serve(grpc_addr);

    info!("========================================");
    info!("🚀 gRPC Server ready!");
    info!("   gRPC endpoint: {}", grpc_addr);
    info!("   JobService: ENABLED (SubmitJob, GetJobStatus, CancelJob, StreamJobUpdates, ListJobs)");
    info!("   NodeService: ENABLED (RegisterNode, Heartbeat, ReportProgress, ReportCompletion)");
    info!("   JobStatusService: ENABLED (UpdateJobStatus, ReportJobResult)");
    info!("   REST API: ENABLED (job submission and status queries now available)");
    info!("========================================");

    // Run gRPC server
    tokio::select! {
        result = grpc_server => {
            if let Err(e) = result {
                error!("gRPC server error: {}", e);
            }
        }
        _ = tokio::signal::ctrl_c() => {
            info!("Received shutdown signal");
        }
    }

    info!("Shutting down gracefully...");
    Ok(())
}
