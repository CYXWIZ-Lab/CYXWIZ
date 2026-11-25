---
title: Run a remote job
description: Reserve a server node in the Server Connection window, train your graph on it, extend, leave and come back, and read how it ended.
type: how-to
area: remote
audience: user
status: experimental
since: 1.0.8
owner: remote
last_reviewed: 2026-11-25
related:
  - index.md
  - data-shipping.md
  - limits.md
  - ../server-and-cloud/server-address.md
  - ../user-guide/training/train-and-watch.md
redirect_from:
  - usage/server-connection.md
---

Use this to train a graph on another machine: a server node on your own
network, or one listed by the server you use.

> **Experimental.** Use remote training only with nodes and servers you
> trust; see [Limits](limits.md). Payments are not live: the window shows
> prices, but nothing is charged.

## Before you start

- The [server address](../server-and-cloud/server-address.md) is set in
  **Preferences > Servers**, and you have a CyxWiz account on that server.
- The graph compiles and trains locally (**Compile**, F7). Its Data Inputs
  feed the Data Loader directly: data preparation nodes cannot run on a node
  (see [Limits](limits.md#graphs)).
- Every Data Input the job trains on has been applied in this session, or
  points at a CyxCloud file. Data must be a table; see
  [How data reaches the node](data-shipping.md).

## The deal: you reserve a block of time

A reservation holds one node for you for the time you choose, 10 minutes to
8 hours. The clock starts when the reservation is made and runs until it
ends, whether you are connected or not; leaving early returns nothing, and
there is no pause. **Extend** adds time to the same block.

## Reserve and train

1. Choose **Deploy > Connect to Server...**. The **Server Connection**
   window opens.
2. Under **Connection Settings**, click **Connect** to reach the central
   server (the server shown there; **Change in Preferences** changes it),
   and sign in with your CyxWiz account if you are not signed in.
3. In **Available Compute Nodes**, pick a node. The row shows its device,
   device memory, price per hour, reputation and region; **Refresh** reloads
   the list, and the filters with **Search** narrow it. The card under the
   table repeats the node's details.
4. In the node's card, set **How long**, **Epochs** and **Batch size**, and
   read the quote: **Price** (per hour, fee included, or **Free** for a node
   rebuilding its reputation), **Held now**, **Your balance after the hold**
   ("Not tracked yet") and **You pay**. Click **Reserve for ...**.
5. The reservation card shows **Time left**, counted from the central
   server's heartbeat ("From the central server, checked N s ago").
6. Under **Training on the node**, click **Connect to node**. The card reads
   **Connected to node** once the link is up.
7. Read **Will this job fit?**: the Engine measures whether the graph and
   its data fit the node's memory. **Data problem** means a Data Input could
   not be read; **Start training** stays off until it loads.
8. Click **Start training**. The Engine sends the graph and serves the
   node its dataset files. Progress shows in the **P2P Training** panel.
9. When the job finishes ("Job finished: ready for the next one"), set
   **Epochs** and **Batch size** again if you like and click **Start new
   training** to run another job in the same reservation.

**Stop training...** asks first; the reservation stays and the node keeps
the job's checkpoints.

## Follow the job and keep the model

The **P2P Training** panel (sidebar, **Network** group) shows the job's
progress, metrics and log, with **Pause**, **Resume**, **Checkpoint** and
**Stop** (which asks first). When the job has finished, set **Save to:** and
click **Export** to download the trained model from the node as a
`.cyxmodel` file (graph, weights, training settings and history). Test it in
the Engine: the test rows never left your computer.

## Extend the reservation

**Extend**, next to the countdown, adds **+15 min**, **+30 min**, **+1
hour** or **+2 hours**, each with its cost. The countdown turns orange under
10 minutes and red under 5, and says what happens to a running job at the
end. The node gets the new end time at once.

## Leave and come back

**Leave node** closes your link to the node, after a question that states
the end time first ("Your reservation ends at 19:37 ..."): the clock keeps
running. A running job is stopped with a checkpoint the node keeps.

After leaving, the reservation is listed under **Your reservation is
running** with its time left and **Reconnect**. Reconnect gets a fresh token
from the central server and puts the card back. The same list appears when
you open the Engine again while a reservation runs. When the time ends, the
row disappears.

**Disconnect from node** (in **Training on the node**) only drops the link
and keeps the card, for a quick reconnect.

## When it ends

- **Time ran out**: the node stops a running job with a checkpoint, and the
  card becomes a receipt (**Time used**, **Paid**: "Nothing charged:
  payments are not live yet", **Jobs**). Resume the job from the **P2P
  Training** panel after reserving a node again. **Reserve again** starts
  over.
- **The central server no longer knows the reservation** (it was restarted,
  or ended it): the card becomes a receipt saying so. If the central server
  stops answering, the countdown line says so after two missed heartbeats
  and keeps counting from its last reply.

## Errors

Errors from reserving, connecting, extending or starting a job show in a
strip at the top of the window (**Dismiss** closes it) and in the Console.

| Message | Meaning |
| --- | --- |
| Sign in to reserve a node. | Not signed in, or the saved session expired |
| Node rejected connection: Invalid or expired auth token | The node and the central server use different P2P secrets, or the reservation is over |
| node '...' is a data preparation step | Prepare the data in the Engine first; see [Limits](limits.md#graphs) |
| dataset '...' is not loaded in the Engine (apply its Data Input first) | Open the Data Input and Apply |
| dataset '...' is empty (0 rows); apply its Data Input again | The last Apply read nothing (the file moved or is unreadable); nothing is sent |
| Could not extend the reservation | The central server refused (the reservation ended, or is not yours) |
| dataset '...' lives in CyxCloud but the Engine has no CyxCloud connection | Open the Cloud Browser while signed in, so the Engine can get the dataset's access token |
| CyxCloud refused an access token for dataset '...' | The gateway's reason follows (no access to the dataset, not signed in) |
| Data Input '...': CyxCloud file '...' is not Arrow IPC or Parquet | Remote nodes train from those two formats only; convert the file first |

## Related

- [How data reaches the node](data-shipping.md)
- [Run a server node](../server-and-cloud/server-node/run-a-server-node.md):
  the same flow from the node's side.
