# Server Connection: reserving and training on a node

The Server Connection tab (Engine, `View > Server Connection`) is where you rent
compute from the network: connect to the central server, sign in, pick a node,
reserve it for a block of time, connect to it, and train your graph there.

## The model: you rent a block of time

A reservation holds one node for you for the time you choose (10 minutes to 8
hours). The price is the node's hourly rate times that time, shown before you
reserve as **Reserved time**. The clock starts when the reservation is created and
runs until it ends, whether you are connected or not. Nothing is returned for
time you do not use. Extending adds minutes to the same block.

This is the same deal as every cloud and GPU marketplace: a machine held for you
is paid for while it is held. Pausing is not offered, because the node's owner
would be working for free.

## Steps

1. **Connect** to the central server (address at the top), then **Sign in** with
   your CyxWiz account (the same one as the website).
2. **Pick a node** in the Available Compute Nodes table. Its device, memory, price
   and reputation are in the row; the card under the table repeats them.
3. **Reserve**: choose the length, epochs and batch size, check the quote
   (price, reserved time, your balance after the hold) and press **Reserve**.
4. **Connect to node**: the card shows the reservation with its countdown
   (from the central server's heartbeat, "checked N s ago") and **Connected to
   node** once the link is up.
5. **Start training**: the Engine measures whether the graph fits the node's
   memory ("Will this job fit?") and sends the job. Progress shows in the P2P
   Training panel. Only graphs whose Data Input feeds the Data Loader directly
   can train on a node; data preparation nodes (tokenizers, vectorizers,
   normalisation, time series, audio) are refused with a message, so prepare the
   data in the Engine first.

## Extend

**Extend** (next to the countdown) adds 15, 30, 60 or 120 minutes, with the cost
of each. The card turns the countdown orange under 10 minutes and red under 5,
and says what happens to a running job at the end. The node gets the new end
time at once.

## Leaving and coming back

**Leave node** closes your link to the node. The clock keeps running: the popup
states the end time first ("Your reservation ends at 19:37 whether you are
connected or not. Leaving does not pause it."). A running job is stopped with a
checkpoint that the node keeps.

After leaving, the reservation is listed under **Your reservation is running**
with "ends at HH:MM (N min left)" and a **Reconnect** button. Reconnect gets a
fresh token from the central server and puts the card back. The same list
appears when you open the Engine again while a reservation is running.

When the time ends, the row disappears and nothing remains to come back to.

**Disconnect from node** (in the Training on the node card) only drops the P2P
link and keeps the card, for a quick reconnect.

## When it ends

- **Time ran out**: the node stops a running job with a checkpoint and the card
  becomes a receipt ("time ran out", time used, jobs started). Resume the job
  from the P2P Training panel after reserving a node again.
- **The central server no longer knows the reservation** (it was restarted or
  ended it on its side): the card becomes a receipt saying so. If the central
  server stops answering, the countdown line turns red after two missed
  heartbeats and keeps counting from its last reply.

## Errors

Errors from reserving, connecting, extending or starting a job are shown in a
red strip at the top of the tab with a Dismiss button, and in the Console.
Common ones:

| Message | Meaning |
| --- | --- |
| Sign in to reserve a node | not signed in, or the saved session expired |
| Node rejected connection: Invalid or expired auth token | the node and the central server use different P2P secrets, or the token's reservation is over |
| node '...' is a data preparation step | the graph needs preparing in the Engine first (see Steps, 5) |
| Could not extend the reservation | the central server refused (ended, or not yours) |
