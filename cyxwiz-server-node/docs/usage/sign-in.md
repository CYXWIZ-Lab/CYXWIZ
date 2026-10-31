# Signing in to the Server Node

The Server Node GUI opens on a sign-in card. Signing in links this machine to
your CyxWiz account so it can be lent to the network; working offline keeps it
local.

## The card

- **Signing in to `host:port` · Change**: the CyxWiz service this node signs in
  with. The address comes from the node config: `network.cyxwiz_server` (the
  one server address, docs/usage/server-address.md) when set, else
  `network.auth_api_url` (default `http://127.0.0.1:3002/api`). Change opens
  the CyxWiz server dialog.
- **Email** and **Password**. The email is remembered between runs; the
  password never is, and it is wiped from memory as soon as the request is sent.
  Enter submits from either field; the eye button shows the password.
- **Sign in** stays off until both fields have text. While the request runs the
  fields lock and the button shows "Signing in…" with a note.
- **Create an account** opens the website of the current service
  (`https://cyxwiz.com/register` for production, your local site for a local
  stack).
- **Work offline** asks you to confirm what it means: local training only,
  nothing is lent, no one can reserve the machine, it earns nothing and does
  not appear in the node list. Sign in later from Account to go online.
- The line under the card names the daemon version and whether the daemon is
  connected.
- The CyxWiz logo sits faintly behind the card and the window's title bar takes
  the same tone (the logo file is `resources/cyxwiz.png`; without it the plain
  theme backdrop shows).

The cursor starts in the empty field: email on a first run, password when the
email was remembered.

## Messages

One message shows at a time, above the fields. It clears when you edit a field
or press Esc. The service's raw text never shows; the node's log has it.

| Message | What happened | What to do |
| --- | --- | --- |
| Can't reach the CyxWiz service at `host:port` | no connection or a timeout | Try again, or Change server |
| That password doesn't match this email | the service refused the password (401) | try again, or reset it on the website; a "Caps Lock is on" hint shows when it is |
| No account with that email | the service knows no such user | check the spelling, or create an account (link in the message) |
| Your sign-in expired | the saved session was no longer accepted, at startup or while running | sign in again; running jobs were not affected |
| The CyxWiz service had a problem (HTTP 5xx) | the service failed | try again in a moment |
| This account is suspended / Too many sign-in attempts | the service said so (403 / 429) | contact support / wait a minute |

## Change the CyxWiz server

One address decides where the node signs in and the central server it
reports to (docs/usage/server-address.md). The dialog shows the address, four
examples you can click (This PC `http://localhost:8000`, Another PC, No front
door, CyxWiz), what kind of address it is, and a row for central and for
sign-in saying whether each answers (asked about half a second after you
stop typing). An address that can't be used is refused with the reason.
**Use this server** saves it as `network.cyxwiz_server` in the config file the
node loaded, so the daemon reports to and renews the sign-in at the same
server after it restarts. Settings > Network edits the same address.

## After sign-in

A card shows the steps: **Signed in** (your email; the session stays signed in
and is renewed in the background), **Registering this machine** (listing its
devices with your account), **Choose what to lend**. Once the machine is
registered the card closes after a few seconds and, when the registration
happened in this session, the Allocation page opens so you can pick devices,
memory and cores and press Apply to go live.

If registering fails, the card says why ("The CyxWiz service didn't accept it:
…" or "couldn't be reached") with **Try again** and **Continue to the
dashboard**. You stay signed in; the node tries again every minute while it
runs.

## Where things are

- Session file (token, node registration, remembered email):
  `%APPDATA%\CyxWiz\server-node\auth.json` on Windows.
- Sign-in service address: `network.cyxwiz_server`, else
  `network.auth_api_url`, in the node config.
- Code: `src/core/sign_in_presentation.{h,cpp}` (every word and state, with
  `tests/test_sign_in_presentation.cpp`), `src/gui/panels/login_panel.*` (the
  drawing), `src/gui/ui_widgets.*` (the shared buttons, links, fields and
  spinner, coloured from the theme).
