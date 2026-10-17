# The webapp answers a photo in one streamed request and holds nothing between requests

A photo is uploaded once, to one endpoint, and the answer streams back in parts: first the preview (the thumbnail with the photo's bean-region mask and box), then the classification. The server keeps no photo, token or session between requests, so any later step on the same photo, such as classifying the whole photo instead, is a new upload. We chose this over a first request that returns a token and a second that asks for the classification by it, because that would mean holding visitors' photos on the server between requests. We also chose it over WebSockets or server-sent events, which would add a protocol for what is a single answer.

## Consequences

- Every pick costs a full classification in one request (about 9–11 s on `yappy`'s GPU), even when the visitor would have stopped at the mask. The single worker is busy for that whole time.
- Streaming needs buffering turned off on every proxy along the way. The app sends `X-Accel-Buffering: no` and our nginx passes that header on to the nginx in front of it.
- A feature that changes the photo's crop before classifying it, such as editing the box (`TODO(crop editing)` in `webapp/static/index.html`), has to be a second upload carrying the edit, not a follow-up on a held photo.

Source: [OPS-7](../ticket_photos_and_answers_optimization.html) D2, D15.
