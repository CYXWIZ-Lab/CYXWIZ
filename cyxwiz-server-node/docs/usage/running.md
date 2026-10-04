# Running a server node

A server node lends a machine's compute to the CyxWiz network. It registers with
the central server, and the Engine sends it training jobs directly (peer to peer)
once a user has reserved it.

It has two programs:

| Program | What it does |
|---|---|
| `cyxwiz-server-daemon.exe` | the node itself: runs jobs, talks to the central server and the Engine |
| `cyxwiz-server-gui.exe` | the window you use to sign in, choose devices and watch jobs; it controls the daemon |

## 1. Build

From the repository root (`D:\Dev\CyxWiz_Engine`), in the node's build tree:

```powershell
cmake --build build-tofix118 --config Release --target cyxwiz-server-daemon cyxwiz-server-gui -- /m:4
```

Binaries land in `build-tofix118\bin\Release\`. Tests:
`cmake --build build-tofix118 --config Release --target test_job_execution_service`
then `build-tofix118\bin\Release\test_job_execution_service.exe`.

## 2. Check the machine first

```powershell
cd build-tofix118\bin\Release
.\cyxwiz-server-daemon.exe --doctor      # can this node take training jobs?
.\cyxwiz-server-daemon.exe --benchmark   # measure throughput on each verified device, save it
```

`--doctor` prints OK / WARN / FAIL per check and exits 0 when the node is ready.
The benchmark result is reported to the central server at registration, so
nodes are ranked by measured speed.

## 3. Run it

What a node needs to agree on with the network:

- **Central server address** (`--central-server=`): `localhost:50051` on the same
  PC; another machine on the LAN uses this PC's address, e.g. `192.168.1.10:50051`.
- **P2P secret** (`--p2p-secret=`): must equal the central server's
  `jwt.p2p_secret`, or the Engine's connection is refused with "Invalid or expired
  auth token".
- **An account** on the website: the GUI signs in with it (web API at
  `http://127.0.0.1:3002/api`), and the node is registered to that owner.

### Start it

```powershell
cd D:\Dev\CyxWiz_Engine\build-tofix118\bin\Release
.\cyxwiz-server-daemon.exe --central-server=localhost:50051 --p2p-secret=<P2P secret>
.\cyxwiz-server-gui.exe     # second window
```

In the GUI: sign in with your website account, choose the devices (and how much
of each) to lend, then **Apply**. Applying registers the node with the central
server and starts its heartbeat; until then the daemon logs "not connected -
waiting for user allocation".

### On another machine on your network

Copy the node build to the machine, then start the daemon with the central
server PC's LAN address for `--central-server` and the same `--p2p-secret`. The central server's gRPC port (50051) must be reachable through
the Windows firewall.

## 4. Options

| Option | Default | Meaning |
|---|---|---|
| `--central-server=ADDR` | `localhost:50051` | central server gRPC address |
| `--p2p-secret=SECRET` | from the config file | verifies the Engine's tokens (central `jwt.p2p_secret`) |
| `--ipc-address=ADDR` | `localhost:50054` | where the GUI connects |
| `--http-port=PORT` | `8082` | REST inference API (`/v1/predict`) |
| `--inference-addr=ADDR` | `0.0.0.0:50057` | gRPC inference service |
| `--config=PATH` | `~/.cyxwiz/daemon.yaml` | config file (the options above override it) |
| `--tls`, `--tls-cert=`, `--tls-key=`, `--tls-ca=`, `--tls-auto` | off | TLS for the gRPC servers |
| `--doctor`, `--benchmark` | | one-shot checks (section 2) |

Ports the daemon opens (from the config file):

| Port | Service |
|---|---|
| 50052 | P2P: the Engine connects here to send jobs |
| 50053 | terminal access |
| 50054 | IPC for the GUI |
| 50055 | node service |
| 50056 | model deployment |
| 50057 | inference gRPC |
| 8082 | inference REST |

## 5. How a job runs (reservation flow)

1. A user reserves the node in the Engine (Server Connection). The central server
   marks it busy and gives the Engine a token naming the reservation, this node and
   when the reservation ends.
2. The Engine connects to port 50052 with that token; the node checks it with the
   P2P secret.
3. Jobs run until the reservation ends. If time runs out mid-job, the node saves a
   checkpoint, stops the job and tells the Engine "Reservation ended"; the job can
   be resumed in a new reservation. Extending the reservation in the Engine moves
   the node's deadline.
4. The node reports each job's result and the end of the reservation to the
   central server.

## 5b. Sign-in

The GUI signs in with your CyxWiz account and hands the token to the daemon
when you click Apply. Tokens last 24 hours; the daemon renews its own copy
through the web API (`auth_api_url` in the daemon config, default
`http://127.0.0.1:3002/api`) when less than 2 hours remain, so a node can run
for days. If renewal fails after the token has expired, sign in again in the
GUI. There is no wallet sign-in.

Live check against a running central server and web API:

```powershell
$env:CYXWIZ_TEST_API = 'http://127.0.0.1:3002/api'
$env:CYXWIZ_TEST_EMAIL = '<account email>'
$env:CYXWIZ_TEST_PASSWORD = '<account password>'
$env:CYXWIZ_TEST_CENTRAL_SERVER = 'localhost:50051'
$env:CYXWIZ_TEST_CENTRAL_JWT_SECRET = '<central server jwt.secret>'
.	est_job_execution_service.exe "[e2e_signin]"
```

## 6. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| Engine: "Invalid or expired auth token" | `--p2p-secret` differs from the central server's `jwt.p2p_secret`, or the node registered under another id (restart the GUI and Apply again) |
| Node never appears in the Engine's list | not registered yet (Apply in the GUI), or it cannot reach the central server address |
| `--doctor` FAIL on a device route | the device's compute route is not verified on this machine; see the Engine's Compute Devices preferences |
| Job refused "Device problem" / "Out of memory" | the node's admission check: the job does not fit or the route is not verified |

## Related

- `docs/usage/` of the central server and of the local ecosystem setup are kept
  with those (private) projects.
