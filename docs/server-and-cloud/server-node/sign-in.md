---
title: Sign in to a server node
description: The server node GUI's sign-in card, its messages, changing the CyxWiz server, and what happens after you sign in.
type: reference
area: server-and-cloud
audience: operator
status: experimental
since: 1.0.8
owner: server-and-cloud
last_reviewed: 2026-11-25
related:
  - run-a-server-node.md
  - settings.md
  - ../server-address.md
redirect_from:
  - cyxwiz-server-node/docs/usage/sign-in.md
---

The server node GUI opens on a sign-in card. Signing in links the machine to
your CyxWiz account so it can be lent to the network; working offline keeps
it local.

## The card

- **Signing in to `host:port` · Change**: the CyxWiz service this node signs
  in with. It comes from the node config: `network.cyxwiz_server` (the
  [server address](../server-address.md)) when set, else
  `network.auth_api_url` (default `http://127.0.0.1:3002/api`). **Change**
  opens the **CyxWiz server** dialog (below).
- **Email** and **Password**. The email is remembered between runs; the
  password never is, and it is wiped from memory as soon as the request is
  sent. Enter submits from either field; the eye button shows the password.
- **Sign in** stays off until both fields have text. While the request runs
  the fields lock and the button reads "Signing in…".
- **Create an account** opens the registration page of the current service's
  website.
- **Work offline** asks you to confirm what it means: local training only,
  nothing is lent, no one can reserve the machine, and it does not appear in
  the node list. Sign in later from **Account** to go online.
- The line under the card names the daemon version and whether the daemon is
  connected.

The cursor starts in the empty field: email on a first run, password when
the email was remembered.

## Messages

One message shows at a time, above the fields. It clears when you edit a
field or press Esc. The service's raw reply never shows; the node's log has
it.

| Message | What happened | What to do |
| --- | --- | --- |
| Can't reach the CyxWiz service at `host:port` | No connection, or a timeout | Try again, or **Change** the server |
| That password doesn't match this email | The service refused the password | Try again, or reset it on the website; a "Caps Lock is on" hint shows when it is |
| No account with that email | The service knows no such user | Check the spelling, or create an account (link in the message) |
| Your sign-in expired | The saved session was no longer accepted, at start or while running | Sign in again; running jobs are not affected |
| The CyxWiz service had a problem | The service failed (HTTP 5xx) | Try again in a moment |
| This account is suspended / Too many sign-in attempts | The service said so | Contact support, or wait a minute |

## Change the CyxWiz server

One address decides where the node signs in and which central server it
reports to. The **CyxWiz server** dialog shows the address, four examples you
can click (**This PC** `http://localhost:8000`, **Another PC**, **No front
door**, **CyxWiz**), what kind of address it is, and a row each for central
and sign-in saying whether it answers (asked about half a second after you
stop typing). An address that cannot be used is refused with the reason.

**Use this server** saves it as `network.cyxwiz_server` in the config file
the node loaded, so the daemon reports to, and renews the sign-in at, the
same server after it restarts. **Settings > Network** edits the same address.

## After you sign in

A card shows the steps: **Signed in** (your email; the session is renewed in
the background), **Registering this machine** (listing its devices with your
account), **Choose what to lend**. Once the machine is registered the card
closes after a few seconds and, when the registration happened in this
session, the **Allocation** page opens so you can pick devices, memory and
cores and press **Apply** to go live.

If registering fails, the card says why ("The CyxWiz service didn't accept
it: …" or "couldn't be reached") with **Try again** and **Continue to the
dashboard**. You stay signed in; the node tries again every minute while it
runs.

## How long a sign-in lasts

The GUI hands your sign-in to the daemon when you press **Apply**. Sign-ins
last 24 hours; the daemon renews its own copy at the sign-in service when
less than 2 hours remain, so a node can run for days. If renewal fails after
the sign-in has expired, sign in again in the GUI. There is no wallet
sign-in.

## Where it is kept

- The session (sign-in, node registration, remembered email):
  `%APPDATA%\CyxWiz\server-node\auth.json` on Windows.
- The sign-in service address: `network.cyxwiz_server`, else
  `network.auth_api_url`, in the node config. See
  [Server node settings](settings.md).
