# Joining a training group

Several Server Nodes can train one model together
(`docs/architecture/distributed-training.md`). A node joins a group's
coordinator, builds the job's model once, and then trains round after round:
it receives the group's model, trains on the rows it is given until the
round's time is up, and sends back the change it made.

This page is for running a node in a group by hand, on your own machines.
Being placed in a group by the CyxWiz platform is not built yet.

## What you need

- The coordinator's address (`HOST:PORT`, default port 50061), from whoever
  runs the group.
- This node's ticket for the group, in a file only you can read.
- The job's data parts at the paths the coordinator hands out. For now a
  group worker reads parts from a path (a shared folder, or the same path on
  every machine); it does not fetch them from CyxCloud yet.
- A verified compute device, as for any training job (`--doctor`).

## Start

```
cyxwiz-server-daemon --group-coordinator=192.168.1.20:50061 --group-ticket-file=D:\cyxwiz\group.ticket
```

Write the ticket to the file without a line end, and keep the file to its
owner (`chmod 600` on Linux and macOS). `--group-ticket=` on the command line
is refused: every user of the machine could read it in the process list.

The node still starts everything else it normally does. **While it is in a
group it takes no other training job**: a job sent to it is answered with
"This node is training in a group and takes no other job until the group
ends". When the group ends the node serves as before.

## What the log says

| Line | Meaning |
| --- | --- |
| `joining the group at ...` | The node is trying to reach the coordinator. It retries, waiting a little longer each time (up to 30 s), until it answers or the node is stopped. |
| `joined group ... as worker N` | The ticket was accepted. |
| `model built (... values exchanged, layout ...); warm-up ...` | The model is built on this node's device and a few steps were timed. The coordinator works out this node's speed from that. |
| `round R: S steps, N samples, loss L, update sent in T s` | One round done and its update accepted. |
| `round R: the update was not taken: ...` | The coordinator refused the update, with the reason. The node goes on with the next round. |
| `Group finished: ...` | The job ended (completed, or a limit was reached). |
| `Removed from the group: ...` | The coordinator took this node out, with the term: for example it is slower than the group allows, or it missed two rounds in a row. Being slower than the group is not a fault of the node. |
| `Could not go on: ...` | This node stopped: the ticket was refused, the model could not be built, the data could not be read, or training failed. The coordinator is told. |

## Stopping

Ctrl+C (or stopping the daemon) ends the round in progress, tells the
coordinator this node is leaving, and shuts the node down. The group goes on
without it if it still has enough members.

## What a group job's graph must be

The node refuses these when it builds the model, with the reason:

- More than one Data Input. The coordinator owns validation.
- A token column that holds text, or a token-tagging graph. A language model
  trains in a group on token windows (the Tokenize step's output), which
  carry their vocabulary with them.
- Labels stored as text. Store class numbers.
- A learning-rate schedule that waits for the validation loss
  (ReduceLROnPlateau). Schedules by epoch or by step are fine; the rate
  follows the group's progress, not this node's own step count.

A graph whose Data Loader has no model seed trains, but its rounds cannot be
reproduced; the log says so.

## Troubleshooting

| Seen | Cause and fix |
| --- | --- |
| `--group-coordinator and --group-ticket-file go together` | Give both or neither. |
| `Group ticket: ...` | The ticket file is missing, empty, or readable by others. |
| `The group did not take this node: this ticket is not one of the group's` | The ticket is not in the coordinator's job, or has a line end or space the job's copy does not. |
| `The group did not take this node: this node built another model than the group's` | This node has another graph or another node version than the group's first node. |
| `data part N is not a path this node can read` | The coordinator handed out a CyxCloud part; group workers read paths only for now. |
| `the leased file could not be read: ...` | The path does not exist on this machine. Use a share every node can reach at the same path. |
| `round_device_changed` | The node's compute device was changed while it was in a group. Restart the node; it joins again and goes on. |
