---
status: accepted
---

# The API is open to pages on any origin

`POST /analyze` answers every origin with `Access-Control-Allow-Origin: *`, so a script on any site's page can send a photo and read the answer, as `curl`, a server or an agent already can. The headers are set by the front's nginx alone, on every answer to `/analyze` including the front's own refusals, and a preflight is answered there without taking the one-at-a-time gate. The app and the check service send none, and the front strips any they would send.

We chose `*` over an allow-list of origins because CORS controls only who may *read* the answer, not who may *send* a photo. A cross-origin `fetch` of a form is a simple request that reaches the worker either way, so an allow-list would keep the answer from a site that can already take the gate. The API takes no key, account or cookie, which is also why `*` is safe: it cannot be combined with credentials, and none are allowed.

## Consequences

- Every caller shares the one worker. A popular site's visitors can make the page answer "busy", and the owner accepts this. The front's access log records each request's `Origin`. That is the evidence a later ticket needs before it narrows access (a per-origin limit, an allow-list or a key), and narrowing would break the sites that use the API by then.
- A key or a cookie would end `*`. Credentials need the exact origin echoed back with `Vary: Origin`, so that change revisits this ADR.
- The contract (`llms.txt`, `openapi.json`) asks a calling site to tell its visitors that their photo is sent here. The photo is never stored, whoever sends it.

Source: [OPS-19](../ticket_cors_analyze.html) D1, D2, D3, D6.
