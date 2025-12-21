mod api;
mod auth;
mod blockchain;
mod cache;
mod config;
mod database;
mod error;
mod pb; // Needed by scheduler for gRPC client
mod scheduler;
mod tui;

use crate::api::grpc::{JobServiceImpl, JobStatusServiceImpl, NodeServiceImpl, NodeDiscoveryServiceImpl, WalletServiceImpl};
use crate::api::rest::v1::infrastructure::InfrastructureState;
use crate::api::rest::create_rest_router;
use crate::auth::middleware::grpc::create_auth_interceptor;
use crate::auth::JWTManager;
use crate::blockchain::{PaymentProcessor, SolanaClient};
use crate::cache::RedisCache;
use crate::config::Config;
use crate::database::{create_pool, run_migrations, MongoClient};
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

    // Initialize MongoDB for user lookups (reads from Website's database)
    info!("Connecting to MongoDB...");
    let mongo_client: Option<Arc<MongoClient>> = match MongoClient::new(&config.mongo).await {
        Ok(client) => {
            info!("✓ MongoDB connected successfully to database: {}", config.mongo.database);
            Some(Arc::new(client))
        }
        Err(e) => {
            warn!("⚠ MongoDB connection failed: {}", e);
            warn!("  User lookups will be unavailable");
            warn!("  Authenticated operations may fail");
            info!("");
            info!("  To enable MongoDB:");
            info!("  1. Configure mongo.url in config.toml with your Atlas connection string");
            info!("  2. Or run local MongoDB: docker run -d -p 27017:27017 mongo:latest");
            info!("");
            None
        }
    };

    // Check command line arguments for mode
    let args: Vec<String> = std::env::args().collect();
    let tui_mode = args.iter().any(|arg| arg == "--tui" || arg == "-t");

    if tui_mode {
        // TUI mode - simpler initialization without payment processor in scheduler
        let tui_scheduler = Arc::new(JobScheduler::new(
            db_pool.clone(),
            redis_cache_arc.read().await.clone(),
            config.scheduler.clone(),
        ));

        // Start scheduler loop in background
        let scheduler_clone = Arc::clone(&tui_scheduler);
        tokio::spawn(async move {
            scheduler_clone.run().await;
        });

        // Build SolanaClient for TUI if keypair exists
        let tui_solana_client = if std::path::Path::new(&config.blockchain.payer_keypair_path).exists() {
            match SolanaClient::from_keypair_file(
                &config.blockchain.solana_rpc_url,
                &config.blockchain.payer_keypair_path,
                &config.blockchain.program_id,
            ) {
                Ok(client) => {
                    info!("TUI: Solana client initialized (network: {})", config.blockchain.network);
                    info!("TUI: Payer pubkey: {}", client.payer_pubkey());
                    Some(Arc::new(client))
                }
                Err(e) => {
                    warn!("TUI: Failed to initialize Solana client: {}", e);
                    warn!("TUI: Blockchain features will be limited");
                    None
                }
            }
        } else {
            warn!("TUI: Solana keypair not found at {}", config.blockchain.payer_keypair_path);
            warn!("TUI: Blockchain features will be limited");
            None
        };

        // TUI mode with scheduler and blockchain
        info!("Starting in TUI mode with live job processing...");
        info!("========================================");
        return tui::run(db_pool, redis_cache_arc, tui_solana_client).await.map_err(|e| e.into());
    }

    // gRPC and REST server mode (default)
    // Initialize Solana blockchain client FIRST (needed for scheduler)
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

    // Initialize payment processor and shared blockchain client
    let (payment_processor, blockchain_client) = if let Some(client) = solana_client {
        // Query real balance from devnet
        match client.get_balance(&client.payer_pubkey()).await {
            Ok(balance) => {
                let sol_balance = balance as f64 / 1_000_000_000.0;
                info!("✓ Payer balance: {:.4} SOL", sol_balance);
                if sol_balance < 0.01 {
                    warn!("⚠ Low balance! Consider requesting an airdrop:");
                    warn!("  solana airdrop 1 {} --url devnet", client.payer_pubkey());
                }
            }
            Err(e) => {
                warn!("⚠ Could not fetch balance: {}", e);
            }
        }

        // Create shared Arc for SolanaClient
        let client_arc = Arc::new(client);
        // Use CYXWIZ devnet config for proper token mint address
        let processor = Arc::new(PaymentProcessor::with_config(
            (*client_arc).clone(),
            crate::blockchain::payment_processor::PaymentConfig::cyxwiz_devnet(),
        ));
        (processor, Some(client_arc))
    } else {
        // No Solana client available - payment features disabled
        // Create a dummy client with invalid keypair (will fail on actual transactions)
        warn!("⚠ Payment processing DISABLED - no valid keypair");
        warn!("  Generate a keypair with: solana-keygen new -o ~/.config/solana/id.json");

        // Generate a dummy 64-byte keypair for initialization only
        // This client will fail on actual transactions but allows server to start
        let dummy_bytes = [0u8; 64];
        let dummy_client = SolanaClient::new(
            "https://api.devnet.solana.com",
            &dummy_bytes,
            &config.blockchain.program_id,
        )?;
        (Arc::new(PaymentProcessor::with_config(
            dummy_client,
            crate::blockchain::payment_processor::PaymentConfig::cyxwiz_devnet(),
        )), None)
    };

    // Initialize job scheduler WITH payment processor for blockchain payments
    info!("Starting job scheduler with payment integration...");
    let scheduler = Arc::new(JobScheduler::with_payment_processor(
        db_pool.clone(),
        redis_cache_arc.read().await.clone(),
        config.scheduler.clone(),
        Arc::clone(&payment_processor),
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

    // Initialize JWT manager for P2P authentication
    info!("Initializing JWT manager...");
    let jwt_manager = Arc::new(JWTManager::new(&config.jwt.secret));
    info!("JWT manager initialized (token expiration: {}s)", config.jwt.p2p_token_expiration_seconds);

    // Initialize gRPC services
    let job_service = JobServiceImpl::new(
        db_pool.clone(),
        Arc::clone(&scheduler),
        Arc::clone(&payment_processor),
        Arc::clone(&jwt_manager),
    );
    info!("Blockchain operations enabled: {}", config.blockchain.enabled);
    let node_service = NodeServiceImpl::new(
        db_pool.clone(),
        Arc::clone(&scheduler),
        Arc::clone(&jwt_manager),
        mongo_client.clone(),
        config.jwt.clone(),
    );
    let job_status_service = JobStatusServiceImpl::new(db_pool.clone(), Arc::clone(&scheduler));
    let wallet_service = WalletServiceImpl::new(blockchain_client.clone());
    let node_discovery_service = NodeDiscoveryServiceImpl::new(db_pool.clone());

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

    // Create auth interceptor
    let auth_interceptor = create_auth_interceptor(Arc::clone(&jwt_manager));

    let grpc_server = Server::builder()
        .add_service(pb::job_service_server::JobServiceServer::with_interceptor(job_service, auth_interceptor.clone()))
        .add_service(pb::node_service_server::NodeServiceServer::with_interceptor(node_service, auth_interceptor.clone()))
        .add_service(pb::node_discovery_service_server::NodeDiscoveryServiceServer::new(node_discovery_service))
        .add_service(pb::job_status_service_server::JobStatusServiceServer::with_interceptor(job_status_service, auth_interceptor.clone()))
        .add_service(pb::wallet_service_server::WalletServiceServer::with_interceptor(wallet_service, auth_interceptor.clone()))
        // .add_service(pb::deployment_service_server::DeploymentServiceServer::new(deployment_service)) // Temporarily disabled
        // .add_service(pb::terminal_service_server::TerminalServiceServer::new(terminal_service)) // Temporarily disabled
        // .add_service(pb::model_service_server::ModelServiceServer::new(model_service)) // Temporarily disabled
        .serve(grpc_addr);

    // Build REST API server
    let rest_addr = "0.0.0.0:8080";
    info!("Starting REST API server on {}", rest_addr);

    let rest_state = InfrastructureState {
        db_pool: db_pool.clone(),
        redis_cache: Some(Arc::clone(&redis_cache_arc)),
        payment_processor: Some(Arc::clone(&payment_processor)),
        scheduler: Some(Arc::clone(&scheduler)),
        start_time: chrono::Utc::now(),
        // Authentication components
        mongo_client: mongo_client.clone(),
        jwt_manager: Some(Arc::clone(&jwt_manager)),
        jwt_config: config.jwt.clone(),
    };

    let rest_router = create_rest_router(rest_state);
    let rest_listener = tokio::net::TcpListener::bind(rest_addr).await?;
    let rest_server = axum::serve(rest_listener, rest_router);

    info!("========================================");
    info!("🚀 Servers ready!");
    info!("   gRPC endpoint: {}", grpc_addr);
    info!("   REST endpoint: http://{}", rest_addr);
    info!("   ");
    info!("   gRPC Services:");
    info!("     - JobService: ENABLED");
    info!("     - NodeService: ENABLED");
    info!("     - JobStatusService: ENABLED");
    info!("   ");
    info!("   REST API v1 Endpoints:");
    info!("     - GET /api/v1/health");
    info!("     - GET /api/v1/infrastructure/stats");
    info!("     - GET /api/v1/nodes/live");
    info!("     - GET /api/v1/nodes/stats");
    info!("     - GET /api/v1/nodes/:id/metrics");
    info!("     - GET /api/v1/scheduler/status");
    info!("     - GET /api/v1/scheduler/queue");
    info!("     - GET /api/v1/scheduler/assignments");
    info!("     - GET /api/v1/scheduler/throughput");
    info!("     - GET /api/v1/blockchain/wallet");
    info!("     - GET /api/v1/blockchain/transactions");
    info!("     - GET /api/v1/blockchain/escrows");
    info!("     - GET /api/v1/blockchain/stats");
    info!("========================================");

    // Run both servers concurrently
    tokio::select! {
        result = grpc_server => {
            if let Err(e) = result {
                error!("gRPC server error: {}", e);
            }
        }
        result = rest_server => {
            if let Err(e) = result {
                error!("REST server error: {}", e);
            }
        }
        _ = tokio::signal::ctrl_c() => {
            info!("Received shutdown signal");
        }
    }

    info!("Shutting down gracefully...");
    Ok(())
}
