  Terminal 1 - Start the daemon:
  # Basic (uses defaults: IPC on localhost:50054)
  cyxwiz-server-daemon

  # With custom options
  cyxwiz-server-daemon --ipc-address=0.0.0.0:50054 --central-server=central.cyxwiz.io:50051

  Terminal 2 - Start the GUI client:
  # GUI mode (default)
  cyxwiz-server-gui

  # TUI mode (btop-style terminal UI)
  cyxwiz-server-gui --tui

  # Connect to remote daemon
  cyxwiz-server-gui --daemon=192.168.1.100:50054

  # Remote daemon with TUI
  cyxwiz-server-gui --daemon=mining-rig.local:50054 --tui

  Remote Management Example

  # On mining rig (headless server)
  cyxwiz-server-daemon --ipc-address=0.0.0.0:50054

  # On your laptop (manage remotely)
  cyxwiz-server-gui --daemon=mining-rig.local:50054

  Running as a System Service

  Linux (systemd):
  # /etc/systemd/system/cyxwiz-daemon.service
  [Unit]
  Description=CyxWiz Server Daemon
  After=network.target

  [Service]
  ExecStart=/usr/bin/cyxwiz-server-daemon --config=/etc/cyxwiz/daemon.yaml
  Restart=always
  User=cyxwiz

  [Install]
  WantedBy=multi-user.target

  sudo systemctl enable cyxwiz-daemon
  sudo systemctl start cyxwiz-daemon

  # Then connect with GUI from anywhere
  cyxwiz-server-gui --daemon=localhost:50054

  Windows (as service):
  # Using NSSM or similar
  nssm install CyxWizDaemon "C:\Program Files\CyxWiz\cyxwiz-server-daemon.exe"
  nssm start CyxWizDaemon

    | Phase 2 | 🔄 In Progress    | Model Management (Model Browser, Deployment, HTTP API, API Keys)      |
  | Phase 3 | ⬜ Not Started     | Network Features (Marketplace, Pool Mining)                           |
  | Phase 4 | ⬜ Not Started     | Security & Polish (TLS, Sandboxing, Wallet Security)                  |
  | Phase 5 | ⬜ Not Started     | Advanced Features (Remote Management, Analytics)