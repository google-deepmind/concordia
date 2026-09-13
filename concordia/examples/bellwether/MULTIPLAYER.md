# A shared Bellwether night

This optional mode replaces **only Nell’s acting policy** with HumanActComponent.
A second person controls Nell while the coordinator makes the twelve watch
choices. Mara, Ivo and Sam keep the selected standard minimal/basic policies.
Each human has an independent existing HumanSession. Standard Sequential still
chooses one entity at a time; there is no separate multiplayer engine.

## Local start

From a source checkout with the Bellwether dependencies installed (see GAME.md):

```shell
python -m concordia.examples.bellwether.run --mode fixture --multiplayer --editor-port 8788 --player-port 8789 --output runs/shared-night
```

`fixture` explicitly scripts the remaining AI residents. Use `--mode live
--model llama3.2:3b` for local Ollama, subject to GAME.md’s model limits. The game
waits for explicit Begin; neither opening a page nor reloading launches a run.
Omit `--multiplayer` to retain single-player behavior.

1. The host opens the local editor on port8788. Do **not** publish this listener.
2. Players open the player page on port8789, enter a name for the host and select
   Coordinator or Nell. A separate spectator browser can request public-only
   access. Use separate browser profiles/devices, not two tabs sharing cookies,
   for separate people.
3. In the editor’s authoritative state, the host finds `join_requests`. After
   confirming the intended person out of band, choose `session.approve`, enter
   that request’s public `id`, and Apply. A browser name is NOT identity proof;
   do not approve an unknown request because it claims a familiar name.
4. After both players are approved, either may Begin. The coordinator can ask
   Nell for fuel and part. Only Nell sees her pending decision control and full
   own context. Nell may Accept, Decline or offer different terms. Acceptance
   records consent, not a completed transfer. The coordinator must separately
   carry out an authorized transfer.
5. Public discussions appear in both journals; private messages appear only to
   their recipients. A spectator sees only events delivered to **all** actors,
   never private accounts or pending human IDs/contexts, and cannot mutate.
6. Reload or close/reopen a tab to reconnect. This browser retains its role and
   the server retains its pending turn; draft text is per-session/per-role.
   Clearing cookies requires a fresh host-approved request. There is no automatic
   AI takeover or simulated action when a player disconnects. If the browser
   reports that it is offline, sending and joining are disabled immediately;
   your current journal and editable draft remain available. On reconnect, the
   page waits for a fresh, authorized snapshot before enabling network actions.
   Returning online does not confirm that the server is reachable and never
   resends an action whose reply was lost. Check the journal before retrying.
7. To move a role to a new device, host `session.revoke` the old request and approve
   the new one. The game continues waiting for the same pending input. Existing
   streams lose authorization on their next delivered snapshot. Already viewed
   data cannot be retracted from a participant’s memory/device.

## HTTPS / Android delivery

This prototype uses established opaque HTTP cookie sessions (Python `secrets`
and `http.cookies`), not URL credentials or untrusted client role claims. Cookies
are HttpOnly and SameSite=Strict; role authorization stays in server memory.
Authenticated snapshots are never cached; SSE re-resolves authorization at each
delivery. Only the trusted host can approve or revoke a role. Sessions and replay
keys last for this process, not a disk-restorable account/checkpoint service.
Admission capacity is128 browser sessions; there is no unbounded guest table or
silent eviction of an existing player.

For a tailnet HTTPS reverse proxy, set `--public-origin https://your-tailnet-host`
and `--cookie-path /bellwether/` on a **new** server invocation. Route only that
player prefix to port8789, stripping the prefix. The page uses relative API URLs.
These options use Secure cookies and check the exact configured Origin for POSTs;
request Forwarded headers do not choose or bypass that origin. A path is not part
of `--public-origin`. Keep the developer listener loopback-only/unproxied.
Inspect and preserve existing routes; no public Funnel is needed or supported.
This is host-approved session authorization on top of your tailnet boundary,
not a claim that a local developer port is remotely authenticated. Never serve
private play over unencrypted remote HTTP. Local testing without public-origin
uses non-Secure cookies only on the default127.0.0.1 listeners.

The touch layout includes a responsive facility map,44px-or-larger player
buttons, visible role/turn status, per-role drafts, a bottom action jump, and
viewport-resizing/scrollable forms for virtual keyboards. Chromium mobile
emulation is not evidence of actual Android/Pixel keyboard behavior; see recorded
validation. No microphone or automatic audio is enabled by this increment.

## Shared operations and boundaries

The host GUI and attached `concordia_session` CLI call the same registered
`session.approve`/`session.revoke` handlers and revision/retry ledger. Player
`human.respond` handlers are chosen by the server-resolved role, not a body field.
The cookie join exchange is authentication setup, not another game mutation API.
A stale or wrong-role submission has no effects. Repeating a successful envelope
from the same authorized browser has one effect. Spectators cannot Begin, respond,
approve roles, or edit components. No private player credentials should be pasted
into a CLI argument, URL, logs or transcript; use the browser join flow.

Limitations: one shared process, two selected human roles, one approved spectator
slot, host-mediated enrollment/reassignment, no lobby directory/accounts, no
cross-server persistence or generalized human-role editor. Arbitrary mechanics,
checkpoint contracts and BRIEF’s full A–H showcase remain separate work.
