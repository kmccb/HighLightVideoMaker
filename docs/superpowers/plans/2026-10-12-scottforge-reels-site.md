# ScottForge Reels Site Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A static site that sells done-for-you highlight reels: sample reel, pricing, one order form, Stripe card payment, and an order email to Tom.

**Architecture:** Static HTML/CSS/JS in `public/`, served by Vercel. Two Vercel functions in `api/` (web-standard `export function POST(request)`), each a thin wrapper over a testable handler in `server/` that takes its dependencies as arguments. Price table and order validation are ES modules in `public/lib/`, imported by both the browser and the functions. No database: Stripe Checkout Session metadata carries the order.

**Tech Stack:** Node 20+, ES modules, `node --test` (built in), `stripe` npm package, Resend REST API via `fetch`, Vercel.

**Spec:** `C:\Users\tom\Claude\HighLightVideoMaker\docs\superpowers\specs\2026-10-12-scottforge-reels-site-design.md`. Read it before starting.

**Repository:** a new one at `C:\Users\tom\Claude\ScottForgeReels`. Nothing in the HighLightVideoMaker repo changes except in Task 10 (sample reels are built with the app, not by editing it).

**Two deviations from the spec's file layout, on purpose:** static files live under `public/` (so server code is never served), and server handlers live in `server/` with `api/` holding only the wiring. One extra setting, `EMAIL_FROM`, is needed because Resend requires a sender address on a verified domain.

## Global Constraints

- Name on the site: **ScottForge Reels**. Contact address everywhere: **tom@scottforge.ai**.
- Prices in US dollars, labelled "Launch price": Basic 39 / 59 / 79 and Spotlight 79 / 119 / 159 for up to 10 / 20 / 30 plays.
- The amount charged is computed on the server from tier and size. Nothing sent by the browser sets it.
- Promises shown wherever an order is summarised: "Ready within 5 days of your clips arriving", "MP4 file and YouTube link", "One round of changes included".
- Refund rule: full refund any time before work starts, none after delivery.
- Song caveat text, verbatim: "You get the reel as an MP4 file and as a YouTube link. YouTube and Instagram sometimes mute or block videos that use copyrighted songs. If that happens, the MP4 is still yours to keep and share."
- Every free-text field is capped at 500 characters (Stripe's limit per metadata value).
- Brand tokens: coal `#0C0B09`, coal-deep `#060505`, ash `#141210`, soot `#1C1915`, steel `#2A2520`, ember `#E85D2F`, ember-hot `#FF8A3D`, ember-yel `#FFC857`, paper `#F2EBDD`, paper-dim `#D4CBB9`, fog `#8A8276`, fog-deep `#554F46`. Fonts: Fraunces (display; the emphasised word is italic ember), Archivo (body), JetBrains Mono (small uppercase labels). Section labels are numbered: "01 / Pricing".
- Copy is plain and first-person plural ("we"). No em dashes in page copy.
- No recruiting package, accounts, uploads, discount codes or add-ons.
- Secrets live only in `.env.local` (git-ignored) and Vercel's settings. Never commit a key, never print one.

## Review Focus

Inputs the spec implies but doesn't spell out. Each has a test in the task that owns the code.

1. A required field that is only spaces, and names with apostrophes or accents (O'Brien, José): spaces are rejected, real names pass. (Task 2)
2. A clips link that isn't a web link (`drive.google.com/x` with no scheme, `javascript:alert(1)`): rejected with a message that says what to paste. (Task 2)
3. A request that tries to set its own price (`amount: 1`) or sends size as the string `"20"`: charged the table price. (Task 3)
4. HTML typed into a field (`<script>`, `<b>`): shown as text in Tom's email, never as markup. (Task 4)
5. A Stripe event that is not a paid order from this site (another event type, an unpaid session, a session with no order metadata): answered 200, no email sent. (Task 5)

---

### Task 1: Repository and price table

**Files:**
- Create: `package.json`, `.gitignore`, `vercel.json`, `public/lib/prices.js`
- Test: `tests/prices.test.js`

**Interfaces:**
- Produces: `TIERS` (`{ basic: 'Basic', spotlight: 'Spotlight' }`), `SIZES` (`[10, 20, 30]`), `price(tier, size) -> number | null` (whole dollars; `size` may be a number or numeric string), `label(tier, size) -> string`.

- [ ] **Step 1: Create the repository**

```bash
mkdir -p /c/Users/tom/Claude/ScottForgeReels && cd /c/Users/tom/Claude/ScottForgeReels && git init -b main && node --version
```
Expected: a version of 20 or higher. If lower, stop and tell Tom Node needs updating.

- [ ] **Step 2: Write `package.json`, `.gitignore`, `vercel.json`**

`package.json`:
```json
{
  "name": "scottforge-reels",
  "private": true,
  "type": "module",
  "engines": { "node": ">=20" },
  "scripts": {
    "test": "node --test",
    "dev": "node --env-file=.env.local dev-server.js"
  }
}
```
`.gitignore`:
```
node_modules/
.env*
.vercel/
```
`vercel.json`:
```json
{ "cleanUrls": true }
```

- [ ] **Step 3: Write the failing test** (`tests/prices.test.js`)

```js
import test from 'node:test';
import assert from 'node:assert/strict';
import { price, label, TIERS, SIZES } from '../public/lib/prices.js';

test('every package has its launch price', () => {
  assert.equal(price('basic', 10), 39);
  assert.equal(price('basic', 20), 59);
  assert.equal(price('basic', 30), 79);
  assert.equal(price('spotlight', 10), 79);
  assert.equal(price('spotlight', 20), 119);
  assert.equal(price('spotlight', 30), 159);
});

test('size may arrive as a string', () => {
  assert.equal(price('spotlight', '20'), 119);
});

test('anything outside the table has no price', () => {
  for (const [t, s] of [['gold', 10], ['basic', 15], ['basic', ''], ['basic', null],
    ['basic', [10]], ['__proto__', 10], ['basic', '20abc'], [undefined, undefined]]) {
    assert.equal(price(t, s), null, `${t} / ${s}`);
  }
});

test('label names the package', () => {
  assert.equal(label('spotlight', 20), 'Spotlight highlight reel, up to 20 plays');
  assert.deepEqual(Object.keys(TIERS), ['basic', 'spotlight']);
  assert.deepEqual(SIZES, [10, 20, 30]);
});
```

- [ ] **Step 4: Run it to see it fail**

Run: `npm test`
Expected: FAIL, cannot find module `public/lib/prices.js`.

- [ ] **Step 5: Implement** (`public/lib/prices.js`)

```js
// Launch prices in whole US dollars. The order page shows these; the checkout function charges them.
export const TIERS = { basic: 'Basic', spotlight: 'Spotlight' };
export const SIZES = [10, 20, 30];

const TABLE = {
  basic: { 10: 39, 20: 59, 30: 79 },
  spotlight: { 10: 79, 20: 119, 30: 159 },
};

export function price(tier, size) {
  if (typeof size !== 'number' && typeof size !== 'string') return null;
  if (typeof tier !== 'string' || !Object.hasOwn(TABLE, tier)) return null;
  const n = Number(size);
  return SIZES.includes(n) && size !== '' ? TABLE[tier][n] : null;
}

export function label(tier, size) {
  return `${TIERS[tier]} highlight reel, up to ${Number(size)} plays`;
}
```

- [ ] **Step 6: Run the tests**

Run: `npm test`
Expected: PASS, 4 tests.

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "Price table and project setup"
```

---

### Task 2: Order validation

**Files:**
- Create: `public/lib/order.js`, `tests/fixtures.js`
- Test: `tests/order.test.js`

**Interfaces:**
- Consumes: `price(tier, size)` from Task 1.
- Produces:
  - `MAX` (500), `SPORTS` (array of strings), `FIELDS` (the text field names, in form order), `CHECKS` (`['song_ok', 'guardian', 'sample']`).
  - `validate(input) -> { ok: true, order } | { ok: false, field, message }`. `order` has `tier`, `size` (number), every name in `FIELDS` as a trimmed string, and the three `CHECKS` as booleans.
  - `toMetadata(order) -> Record<string, string>` and `fromMetadata(meta) -> order | null` (null when `meta` has no valid tier and size).

- [ ] **Step 1: Write the fixture and the failing tests**

`tests/fixtures.js` (a complete, valid order; later tasks' tests use it too):
```js
export const good = () => ({
  tier: 'spotlight', size: '20',
  first: 'Caleb', last: 'Scott', number: '12', position: 'WR / DB',
  school: 'Poland Bulldogs', grad: '2027', sport: 'Football',
  colour1: 'Navy', colour2: 'White', spot: 'White cleats, lines up wide left',
  clips: 'https://drive.google.com/drive/folders/abc', song: 'Song - Artist',
  name: 'Tom Scott', email: 'parent@example.com', notes: '',
  song_ok: true, guardian: true, sample: false,
});
```

`tests/order.test.js`:

```js
import test from 'node:test';
import assert from 'node:assert/strict';
import { validate, toMetadata, fromMetadata, MAX, FIELDS, CHECKS } from '../public/lib/order.js';
import { good } from './fixtures.js';

const bad = (change, field) => {
  const v = validate({ ...good(), ...change });
  assert.equal(v.ok, false, JSON.stringify(change));
  assert.equal(v.field, field);
  assert.ok(v.message.length > 5);
};

test('a complete order passes and is cleaned up', () => {
  const v = validate({ ...good(), first: '  Caleb ' });
  assert.equal(v.ok, true);
  assert.equal(v.order.first, 'Caleb');
  assert.equal(v.order.size, 20);
  assert.equal(v.order.sample, false);
});

test('required fields', () => {
  for (const f of ['first', 'last', 'sport', 'clips', 'song', 'name', 'email']) bad({ [f]: '' }, f);
});

test('a field of only spaces counts as empty', () => {
  bad({ first: '   ' }, 'first');
  bad({ song: '\t\n' }, 'song');
});

test('real names pass', () => {
  for (const last of ["O'Brien", 'José', 'Smith-Jones', 'Nguyễn']) {
    assert.equal(validate({ ...good(), last }).ok, true, last);
  }
});

test('package must be in the price table', () => {
  bad({ tier: 'gold' }, 'tier');
  bad({ size: '15' }, 'tier');
  bad({ size: undefined }, 'tier');
});

test('Spotlight needs a jersey number, Basic does not', () => {
  bad({ number: '' }, 'number');
  const v = validate({ ...good(), tier: 'basic', number: '' });
  assert.equal(v.ok, true);
  assert.equal(v.order.spot, '', 'the spot-them note only applies to Spotlight');
});

test('clips must be a web link', () => {
  bad({ clips: 'drive.google.com/x' }, 'clips');
  bad({ clips: 'javascript:alert(1)' }, 'clips');
  bad({ clips: 'my google drive' }, 'clips');
  assert.match(validate({ ...good(), clips: 'drive.google.com/x' }).message, /https:\/\//);
});

test('email, sport and tick boxes', () => {
  bad({ email: 'not-an-email' }, 'email');
  bad({ sport: 'Quidditch' }, 'sport');
  bad({ song_ok: false }, 'song_ok');
  bad({ guardian: 'true' }, 'guardian');
});

test('length cap and wrong types', () => {
  bad({ notes: 'x'.repeat(MAX + 1) }, 'notes');
  assert.equal(validate({ ...good(), notes: 'x'.repeat(MAX) }).ok, true);
  bad({ first: { toString: () => 'x' } }, 'first');
  assert.equal(validate(null).ok, false);
  assert.equal(validate([]).ok, false);
});

test('metadata round trip', () => {
  const { order } = validate(good());
  const meta = toMetadata(order);
  for (const v of Object.values(meta)) assert.equal(typeof v, 'string');
  assert.ok(Object.keys(meta).length <= 50);
  assert.deepEqual(fromMetadata(meta), order);
  assert.equal(fromMetadata({}), null);
  assert.equal(fromMetadata({ tier: 'gold', size: '10' }), null);
  assert.equal(FIELDS.length + CHECKS.length + 2, Object.keys(meta).length);
});
```

- [ ] **Step 2: Run to see them fail**

Run: `npm test`
Expected: FAIL, cannot find module `public/lib/order.js`.

- [ ] **Step 3: Implement** (`public/lib/order.js`)

```js
// What an order is and when it's complete. Runs in the browser (order page) and on the server (checkout).
import { price } from './prices.js';

export const MAX = 500; // Stripe's limit for one stored value
export const SPORTS = ['Football', 'Basketball', 'Baseball', 'Softball', 'Soccer', 'Lacrosse',
  'Hockey', 'Volleyball', 'Wrestling', 'Track & Field', 'Other'];
export const FIELDS = ['first', 'last', 'number', 'position', 'school', 'grad', 'sport',
  'colour1', 'colour2', 'spot', 'clips', 'song', 'name', 'email', 'notes'];
export const CHECKS = ['song_ok', 'guardian', 'sample'];

const REQUIRED = {
  first: 'First name', last: 'Last name', sport: 'Sport', clips: 'A link to the clips',
  song: 'A song', name: 'Your name', email: 'Your email',
};
const fail = (field, message) => ({ ok: false, field, message });

function isWebLink(s) {
  try {
    const u = new URL(s);
    return u.protocol === 'https:' || u.protocol === 'http:';
  } catch {
    return false;
  }
}

export function validate(input) {
  if (!input || typeof input !== 'object' || Array.isArray(input)) {
    return fail('form', 'Order details are missing.');
  }
  const tier = typeof input.tier === 'string' ? input.tier : '';
  if (price(tier, input.size) === null) return fail('tier', 'Pick a package and a size.');
  const order = { tier, size: Number(input.size) };

  for (const k of FIELDS) {
    const v = input[k] ?? '';
    if (typeof v !== 'string') return fail(k, "That doesn't look right.");
    const t = v.trim();
    if (t.length > MAX) return fail(k, `Keep this under ${MAX} characters.`);
    order[k] = t;
  }
  if (tier !== 'spotlight') order.spot = '';

  for (const [k, name] of Object.entries(REQUIRED)) {
    if (!order[k]) return fail(k, `${name} is needed.`);
  }
  if (tier === 'spotlight' && !order.number) {
    return fail('number', 'Jersey number is needed so we can highlight them.');
  }
  if (!SPORTS.includes(order.sport)) return fail('sport', 'Pick a sport from the list.');
  if (!isWebLink(order.clips)) {
    return fail('clips', 'Paste the full shared link, starting with https://');
  }
  if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(order.email)) return fail('email', 'Check the email address.');
  if (input.song_ok !== true) return fail('song_ok', 'Please tick the box about the song.');
  if (input.guardian !== true) return fail('guardian', 'Please confirm you can order for this player.');

  order.song_ok = true;
  order.guardian = true;
  order.sample = input.sample === true;
  return { ok: true, order };
}

export function toMetadata(order) {
  return Object.fromEntries(Object.entries(order).map(([k, v]) => [k, String(v)]));
}

export function fromMetadata(meta) {
  if (!meta || price(meta.tier, meta.size) === null) return null;
  const order = { tier: meta.tier, size: Number(meta.size) };
  for (const k of FIELDS) order[k] = meta[k] ?? '';
  for (const k of CHECKS) order[k] = meta[k] === 'true';
  return order;
}
```

- [ ] **Step 4: Run the tests**

Run: `npm test`
Expected: PASS, all tests in both files.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "Order validation shared by the page and the server"
```

---

### Task 3: Checkout function

**Files:**
- Create: `server/checkout.js`, `api/checkout.js`
- Test: `tests/checkout.test.js`

**Interfaces:**
- Consumes: `validate`, `toMetadata` (Task 2); `price`, `label` (Task 1).
- Produces: `handleCheckout(request: Request, { stripe, siteUrl }) -> Promise<Response>`. 200 `{ url }` on success; 400 `{ field, message }` for a bad order; 502 `{ field: 'form', message }` when Stripe fails. `stripe` is anything with `checkout.sessions.create(params)`.

- [ ] **Step 1: Install Stripe's library**

Run: `npm install stripe`
Expected: `stripe` appears under `dependencies` in `package.json`.

- [ ] **Step 2: Write the failing tests** (`tests/checkout.test.js`)

```js
import test from 'node:test';
import assert from 'node:assert/strict';
import { handleCheckout } from '../server/checkout.js';
import { good } from './fixtures.js';

const post = (body) => new Request('https://x.test/api/checkout', {
  method: 'POST', headers: { 'content-type': 'application/json' },
  body: typeof body === 'string' ? body : JSON.stringify(body),
});
function fakeStripe(result = { url: 'https://checkout.stripe.test/s/1' }) {
  const calls = [];
  return {
    calls,
    checkout: { sessions: { create: async (p) => {
      calls.push(p);
      if (result instanceof Error) throw result;
      return result;
    } } },
  };
}
const deps = (stripe) => ({ stripe, siteUrl: 'https://forgedreels.com' });

test('a good order opens a Stripe session at the table price', async () => {
  const stripe = fakeStripe();
  const res = await handleCheckout(post(good()), deps(stripe));
  assert.equal(res.status, 200);
  assert.deepEqual(await res.json(), { url: 'https://checkout.stripe.test/s/1' });
  const p = stripe.calls[0];
  assert.equal(p.mode, 'payment');
  assert.equal(p.line_items.length, 1);
  assert.equal(p.line_items[0].quantity, 1);
  assert.equal(p.line_items[0].price_data.currency, 'usd');
  assert.equal(p.line_items[0].price_data.unit_amount, 11900);
  assert.equal(p.line_items[0].price_data.product_data.name, 'Spotlight highlight reel, up to 20 plays');
  assert.equal(p.customer_email, 'parent@example.com');
  assert.equal(p.metadata.first, 'Caleb');
  assert.equal(p.metadata.size, '20');
  assert.equal(p.success_url, 'https://forgedreels.com/thanks');
  assert.equal(p.cancel_url, 'https://forgedreels.com/order?cancelled=1');
});

test('the browser cannot set the price', async () => {
  const stripe = fakeStripe();
  await handleCheckout(post({ ...good(), amount: 1, unit_amount: 1, price: 1, tier: 'basic', size: 10 }), deps(stripe));
  assert.equal(stripe.calls[0].line_items[0].price_data.unit_amount, 3900);
  assert.equal(stripe.calls[0].metadata.amount, undefined);
});

test('a bad order is refused before Stripe is called', async () => {
  const stripe = fakeStripe();
  const res = await handleCheckout(post({ ...good(), email: 'nope' }), deps(stripe));
  assert.equal(res.status, 400);
  assert.equal((await res.json()).field, 'email');
  assert.equal(stripe.calls.length, 0);
});

test('a body that is not JSON is a 400, not a crash', async () => {
  for (const body of ['', 'not json', '[]', 'null']) {
    const res = await handleCheckout(post(body), deps(fakeStripe()));
    assert.equal(res.status, 400, JSON.stringify(body));
    assert.equal((await res.json()).field, 'form');
  }
});

test('Stripe failing gives the customer a way forward', async () => {
  const res = await handleCheckout(post(good()), deps(fakeStripe(new Error('stripe down'))));
  assert.equal(res.status, 502);
  const out = await res.json();
  assert.equal(out.field, 'form');
  assert.match(out.message, /tom@scottforge\.ai/);
});
```

- [ ] **Step 3: Run to see them fail**

Run: `npm test`
Expected: FAIL, cannot find module `server/checkout.js`.

- [ ] **Step 4: Implement**

`server/checkout.js`:
```js
import { validate, toMetadata } from '../public/lib/order.js';
import { price, label } from '../public/lib/prices.js';

const json = (status, body) => new Response(JSON.stringify(body), {
  status, headers: { 'content-type': 'application/json' },
});

export async function handleCheckout(request, { stripe, siteUrl }) {
  let input;
  try {
    input = await request.json();
  } catch {
    input = null;
  }
  const v = validate(input);
  if (!v.ok) return json(400, { field: v.field, message: v.message });
  const { order } = v;

  try {
    const session = await stripe.checkout.sessions.create({
      mode: 'payment',
      line_items: [{
        quantity: 1,
        price_data: {
          currency: 'usd',
          unit_amount: price(order.tier, order.size) * 100,
          product_data: { name: label(order.tier, order.size) },
        },
      }],
      customer_email: order.email,
      metadata: toMetadata(order),
      success_url: `${siteUrl}/thanks`,
      cancel_url: `${siteUrl}/order?cancelled=1`,
    });
    return json(200, { url: session.url });
  } catch (err) {
    console.error('checkout: Stripe could not create a session:', err.message);
    return json(502, {
      field: 'form',
      message: "Payment couldn't start. Try again, or email us at tom@scottforge.ai.",
    });
  }
}
```

`api/checkout.js`:
```js
import Stripe from 'stripe';
import { handleCheckout } from '../server/checkout.js';

export function POST(request) {
  return handleCheckout(request, {
    stripe: new Stripe(process.env.STRIPE_SECRET_KEY),
    siteUrl: process.env.SITE_URL,
  });
}
```

- [ ] **Step 5: Run the tests**

Run: `npm test`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "Checkout function: validate, price on the server, open Stripe"
```

---

### Task 4: Order emails

**Files:**
- Create: `server/email.js`
- Test: `tests/email.test.js`

**Interfaces:**
- Consumes: `TIERS` (Task 1); an `order` as produced by `validate` / `fromMetadata` (Task 2).
- Produces:
  - `esc(value) -> string` (HTML-escapes).
  - `orderEmails(order, { paid, to, from }) -> [toTom, toCustomer]`; each is `{ from, to, reply_to, subject, html, text }`. `paid` is whole dollars.
  - `send(message, { apiKey, fetch? }) -> Promise<void>`; throws if Resend answers non-2xx.

- [ ] **Step 1: Write the failing tests** (`tests/email.test.js`)

```js
import test from 'node:test';
import assert from 'node:assert/strict';
import { esc, orderEmails, send } from '../server/email.js';
import { validate } from '../public/lib/order.js';
import { good } from './fixtures.js';

const order = (change = {}) => validate({ ...good(), ...change }).order;
const opts = { paid: 119, to: 'tom@scottforge.ai', from: 'ScottForge Reels <orders@forgedreels.com>' };

test('Tom gets everything needed to make the reel', () => {
  const [tom] = orderEmails(order(), opts);
  assert.equal(tom.to, 'tom@scottforge.ai');
  assert.equal(tom.reply_to, 'parent@example.com');
  assert.equal(tom.subject, 'New reel order: Caleb Scott #12, Spotlight, up to 20 plays ($119)');
  for (const piece of ['Caleb Scott', '12', 'WR / DB', 'Poland Bulldogs', '2027', 'Football', 'Navy', 'White',
    'White cleats', 'https://drive.google.com/drive/folders/abc', 'Song - Artist', 'Tom Scott',
    'parent@example.com', '$119']) {
    assert.ok(tom.text.includes(piece), `text has ${piece}`);
    assert.ok(tom.html.includes(esc(piece)), `html has ${piece}`);
  }
  assert.ok(tom.text.includes('May show as a sample: No'));
});

test('the customer gets a confirmation that replies go to Tom', () => {
  const [, customer] = orderEmails(order(), opts);
  assert.equal(customer.to, 'parent@example.com');
  assert.equal(customer.reply_to, 'tom@scottforge.ai');
  assert.match(customer.subject, /Caleb/);
  assert.match(customer.text, /within 5 days/);
  assert.match(customer.text, /One round of changes/);
  assert.match(customer.text, /https:\/\/drive\.google\.com\/drive\/folders\/abc/);
  assert.match(customer.text, /tom@scottforge\.ai/);
});

test('anything typed into the form is shown as text, never as markup', () => {
  const [tom, customer] = orderEmails(order({ first: '<script>alert(1)</script>', notes: '<b>bold</b> & "quotes"' }), opts);
  for (const m of [tom, customer]) {
    assert.ok(!m.html.includes('<script>'));
    assert.ok(!m.html.includes('<b>bold</b>'));
  }
  assert.ok(tom.html.includes('&lt;b&gt;bold&lt;/b&gt; &amp; &quot;quotes&quot;'));
  assert.ok(!tom.subject.includes('\n'));
});

test('a subject cannot be split by a line break in a name', () => {
  const o = { ...order(), first: 'Caleb\r\nBcc: x@y.z' };
  assert.ok(!/[\r\n]/.test(orderEmails(o, opts)[0].subject));
});

test('send posts to Resend and fails loudly when refused', async () => {
  const calls = [];
  const ok = async (url, init) => { calls.push([url, init]); return new Response('{"id":"1"}', { status: 200 }); };
  const [tom] = orderEmails(order(), opts);
  await send(tom, { apiKey: 'k', fetch: ok });
  assert.equal(calls[0][0], 'https://api.resend.com/emails');
  assert.equal(calls[0][1].method, 'POST');
  assert.equal(calls[0][1].headers.authorization, 'Bearer k');
  assert.equal(JSON.parse(calls[0][1].body).subject, tom.subject);
  const refused = async () => new Response('domain not verified', { status: 403 });
  await assert.rejects(send(tom, { apiKey: 'k', fetch: refused }), /403/);
});
```

- [ ] **Step 2: Run to see them fail**

Run: `npm test`
Expected: FAIL, cannot find module `server/email.js`.

- [ ] **Step 3: Implement** (`server/email.js`)

```js
import { TIERS } from '../public/lib/prices.js';

const ENTITIES = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };
export const esc = (v) => String(v).replace(/[&<>"']/g, (c) => ENTITIES[c]);
const oneLine = (s) => String(s).replace(/[\r\n]+/g, ' ');

function rows(o, paid) {
  return [
    ['Package', `${TIERS[o.tier]}, up to ${o.size} plays`],
    ['Paid', `$${paid}`],
    ['Player', `${o.first} ${o.last}`],
    ['Jersey number', o.number],
    ['Position', o.position],
    ['School or team', o.school],
    ['Graduation year', o.grad],
    ['Sport', o.sport],
    ['Team colours', [o.colour1, o.colour2].filter(Boolean).join(', ')],
    ['How to spot them', o.spot],
    ['Clips', o.clips],
    ['Song', o.song],
    ['Ordered by', `${o.name} <${o.email}>`],
    ['Notes', o.notes],
    ['May show as a sample', o.sample ? 'Yes' : 'No'],
  ].filter(([, v]) => v !== '');
}

const asText = (list) => list.map(([k, v]) => `${k}: ${v}`).join('\n');
const asHtml = (list) => `<table cellpadding="6" style="font:14px/1.4 Arial,sans-serif;border-collapse:collapse">${
  list.map(([k, v]) => `<tr><td style="color:#666;vertical-align:top">${esc(k)}</td><td>${esc(v)}</td></tr>`).join('')
}</table>`;

export function orderEmails(order, { paid, to, from }) {
  const pack = `${TIERS[order.tier]}, up to ${order.size} plays`;
  const number = order.number ? ` #${order.number}` : '';
  const list = rows(order, paid);

  const toTom = {
    from, to, reply_to: order.email,
    subject: oneLine(`New reel order: ${order.first} ${order.last}${number}, ${pack} ($${paid})`),
    text: asText(list),
    html: asHtml(list),
  };

  const lines = [
    `Thanks, ${order.name}. We've got your order for ${order.first}'s highlight reel.`,
    `Package: ${pack} ($${paid})`,
    `What happens next:`,
    `1. Check that this link opens for anyone who has it: ${order.clips}`,
    `2. We build the reel. It will be ready within 5 days of your clips arriving.`,
    `3. You get an MP4 file and a YouTube link by email. One round of changes is included.`,
    `Questions or something custom? Reply to this email or write to tom@scottforge.ai.`,
    `ScottForge Reels`,
  ];
  const toCustomer = {
    from, to: order.email, reply_to: to,
    subject: oneLine(`We've got your order for ${order.first}'s highlight reel`),
    text: lines.join('\n\n'),
    html: `<div style="font:15px/1.5 Arial,sans-serif">${lines.map((l) => `<p>${esc(l)}</p>`).join('')}</div>`,
  };
  return [toTom, toCustomer];
}

export async function send(message, { apiKey, fetch = globalThis.fetch }) {
  const res = await fetch('https://api.resend.com/emails', {
    method: 'POST',
    headers: { authorization: `Bearer ${apiKey}`, 'content-type': 'application/json' },
    body: JSON.stringify(message),
  });
  if (!res.ok) throw new Error(`email not sent: ${res.status} ${await res.text()}`);
}
```

- [ ] **Step 4: Run the tests**

Run: `npm test`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "Order and confirmation emails"
```

---

### Task 5: Stripe webhook

**Files:**
- Create: `server/webhook.js`, `api/stripe-webhook.js`
- Test: `tests/webhook.test.js`

**Interfaces:**
- Consumes: `fromMetadata` (Task 2); `orderEmails`, `send` (Task 4).
- Produces: `handleWebhook(request: Request, { stripe, secret, sendEmail, to, from }) -> Promise<Response>`. `stripe` is anything with `webhooks.constructEvent(body, signature, secret)` that throws on a bad signature. `sendEmail(message) -> Promise<void>`. 400 on a bad signature; 500 if an email fails (so Stripe retries); 200 otherwise.

- [ ] **Step 1: Write the failing tests** (`tests/webhook.test.js`)

```js
import test from 'node:test';
import assert from 'node:assert/strict';
import { handleWebhook } from '../server/webhook.js';
import { validate, toMetadata } from '../public/lib/order.js';
import { good } from './fixtures.js';

const stripe = { webhooks: { constructEvent: (body, sig, secret) => {
  if (sig !== 'good-sig' || secret !== 'whsec') throw new Error('bad signature');
  return JSON.parse(body);
} } };
const paidEvent = (change = {}) => ({
  type: 'checkout.session.completed',
  data: { object: {
    payment_status: 'paid', amount_total: 11900,
    metadata: toMetadata(validate(good()).order), ...change,
  } },
});
const post = (event, sig = 'good-sig') => new Request('https://x.test/api/stripe-webhook', {
  method: 'POST', headers: sig ? { 'stripe-signature': sig } : {}, body: JSON.stringify(event),
});
function run(event, { sig, failOn } = {}) {
  const sent = [];
  const sendEmail = async (m) => {
    if (failOn === sent.length) throw new Error('email not sent: 500');
    sent.push(m);
  };
  const res = handleWebhook(post(event, sig), {
    stripe, secret: 'whsec', sendEmail, to: 'tom@scottforge.ai', from: 'ScottForge Reels <orders@forgedreels.com>',
  });
  return res.then((r) => ({ status: r.status, sent }));
}

test('a paid order emails Tom and the customer', async () => {
  const { status, sent } = await run(paidEvent());
  assert.equal(status, 200);
  assert.equal(sent.length, 2);
  assert.equal(sent[0].to, 'tom@scottforge.ai');
  assert.match(sent[0].subject, /\$119/);
  assert.equal(sent[1].to, 'parent@example.com');
});

test('a wrong or missing signature is refused and nothing is sent', async () => {
  for (const sig of ['forged', null]) {
    const { status, sent } = await run(paidEvent(), { sig });
    assert.equal(status, 400);
    assert.equal(sent.length, 0);
  }
});

test('events that are not a paid order from this site are acknowledged and ignored', async () => {
  const others = [
    { type: 'payment_intent.succeeded', data: { object: {} } },
    paidEvent({ payment_status: 'unpaid' }),
    paidEvent({ metadata: {} }),
    paidEvent({ metadata: null }),
  ];
  for (const event of others) {
    const { status, sent } = await run(event);
    assert.equal(status, 200, event.type);
    assert.equal(sent.length, 0);
  }
});

test('if an email fails, Stripe is told to try again', async () => {
  assert.equal((await run(paidEvent(), { failOn: 0 })).status, 500);
  assert.equal((await run(paidEvent(), { failOn: 1 })).status, 500);
});
```

- [ ] **Step 2: Run to see them fail**

Run: `npm test`
Expected: FAIL, cannot find module `server/webhook.js`.

- [ ] **Step 3: Implement**

`server/webhook.js`:
```js
import { fromMetadata } from '../public/lib/order.js';
import { orderEmails } from './email.js';

const reply = (status, text) => new Response(text, { status });

export async function handleWebhook(request, { stripe, secret, sendEmail, to, from }) {
  const body = await request.text(); // the signature is over the exact bytes Stripe sent
  let event;
  try {
    event = stripe.webhooks.constructEvent(body, request.headers.get('stripe-signature') ?? '', secret);
  } catch {
    return reply(400, 'bad signature');
  }
  if (event.type !== 'checkout.session.completed') return reply(200, 'ignored');
  const session = event.data.object;
  if (session.payment_status !== 'paid') return reply(200, 'not paid yet');
  const order = fromMetadata(session.metadata);
  if (!order) return reply(200, 'not an order from this site');

  try {
    for (const message of orderEmails(order, { paid: session.amount_total / 100, to, from })) {
      await sendEmail(message);
    }
  } catch (err) {
    console.error('webhook: order email failed, Stripe will retry:', err.message);
    return reply(500, 'email failed');
  }
  return reply(200, 'ok');
}
```

`api/stripe-webhook.js`:
```js
import Stripe from 'stripe';
import { handleWebhook } from '../server/webhook.js';
import { send } from '../server/email.js';

export function POST(request) {
  return handleWebhook(request, {
    stripe: new Stripe(process.env.STRIPE_SECRET_KEY),
    secret: process.env.STRIPE_WEBHOOK_SECRET,
    sendEmail: (message) => send(message, { apiKey: process.env.RESEND_API_KEY }),
    to: process.env.ORDER_EMAIL_TO,
    from: process.env.EMAIL_FROM,
  });
}
```

- [ ] **Step 4: Run the tests**

Run: `npm test`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "Stripe webhook: verify, then email the order"
```

---

### Task 6: Local dev server

Vercel's own dev command needs a signed-in account. This small server serves `public/` with clean URLs and runs the two functions, so pages can be previewed and the order flow tried locally.

**Files:**
- Create: `dev-server.js`, `.env.example`, `.claude/launch.json`, `public/index.html` (one-line stand-in, replaced in Task 8)
- Test: `tests/dev-server.test.js`

**Interfaces:**
- Consumes: `api/checkout.js`, `api/stripe-webhook.js` (their `POST` exports).
- Produces: `start(port) -> Promise<http.Server>`; `npm run dev` on port 3210.

- [ ] **Step 1: Write the failing test** (`tests/dev-server.test.js`)

```js
import test from 'node:test';
import assert from 'node:assert/strict';
import { start } from '../dev-server.js';

test('serves pages with clean URLs and runs the functions', async (t) => {
  const server = await start(0);
  t.after(() => server.close());
  const base = `http://127.0.0.1:${server.address().port}`;

  const home = await fetch(`${base}/`);
  assert.equal(home.status, 200);
  assert.match(home.headers.get('content-type'), /text\/html/);
  assert.equal((await fetch(`${base}/index`)).status, 200);
  assert.equal((await fetch(`${base}/lib/prices.js`)).headers.get('content-type'), 'text/javascript');
  assert.equal((await fetch(`${base}/no-such-page`)).status, 404);
  assert.equal((await fetch(`${base}/..%2Fpackage.json`)).status, 404);
  assert.equal((await fetch(`${base}/api/checkout`)).status, 405);

  const bad = await fetch(`${base}/api/checkout`, { method: 'POST', body: '{}' });
  assert.equal(bad.status, 400);
  assert.equal((await bad.json()).field, 'tier');
});
```

- [ ] **Step 2: Run to see it fail**

Run: `npm test`
Expected: FAIL, cannot find module `dev-server.js`.

- [ ] **Step 3: Implement**

`public/index.html`: the single line `<!doctype html><title>ScottForge Reels</title>`.

`.env.example`:
```
# Copy to .env.local and fill in. Test-mode keys only on this machine.
STRIPE_SECRET_KEY=
STRIPE_WEBHOOK_SECRET=
RESEND_API_KEY=
ORDER_EMAIL_TO=tom@scottforge.ai
EMAIL_FROM=ScottForge Reels <orders@forgedreels.com>
SITE_URL=http://localhost:3210
```

`.claude/launch.json`:
```json
{
  "version": "0.0.1",
  "configurations": [
    { "name": "site", "runtimeExecutable": "npm", "runtimeArgs": ["run", "dev"], "port": 3210 }
  ]
}
```

`dev-server.js`:
```js
// Local stand-in for Vercel: static files from public/ with clean URLs, plus the api/ functions.
import http from 'node:http';
import path from 'node:path';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.join(here, 'public');
const TYPES = {
  '.html': 'text/html; charset=utf-8', '.css': 'text/css', '.js': 'text/javascript',
  '.mp4': 'video/mp4', '.svg': 'image/svg+xml', '.png': 'image/png', '.jpg': 'image/jpeg',
  '.woff2': 'font/woff2', '.ico': 'image/x-icon',
};
const API = { '/api/checkout': './api/checkout.js', '/api/stripe-webhook': './api/stripe-webhook.js' };

async function runFunction(file, req, url) {
  const handler = (await import(file))[req.method];
  if (!handler) return new Response('Method not allowed', { status: 405 });
  const chunks = [];
  for await (const c of req) chunks.push(c);
  const headers = Object.entries(req.headers).filter(([, v]) => typeof v === 'string');
  return handler(new Request(url, {
    method: req.method, headers, body: chunks.length ? Buffer.concat(chunks) : undefined,
  }));
}

export function start(port) {
  const server = http.createServer(async (req, res) => {
    try {
      const url = new URL(req.url, `http://${req.headers.host}`);
      if (API[url.pathname]) {
        const out = await runFunction(API[url.pathname], req, url);
        res.writeHead(out.status, Object.fromEntries(out.headers));
        res.end(Buffer.from(await out.arrayBuffer()));
        return;
      }
      let p = decodeURIComponent(url.pathname);
      if (p.endsWith('/')) p += 'index';
      if (!path.extname(p)) p += '.html';
      const file = path.join(root, p);
      if (!file.startsWith(root + path.sep)) throw Object.assign(new Error('outside public'), { code: 'ENOENT' });
      const data = await readFile(file);
      res.writeHead(200, { 'content-type': TYPES[path.extname(file)] ?? 'application/octet-stream' });
      res.end(data);
    } catch (err) {
      if (err.code === 'ENOENT' || err.code === 'EISDIR') {
        res.writeHead(404).end('Not found');
      } else {
        console.error(err);
        res.writeHead(500).end('Server error');
      }
    }
  });
  return new Promise((resolve) => server.listen(port, () => resolve(server)));
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const port = Number(process.env.PORT) || 3210;
  await start(port);
  console.log(`ScottForge Reels (dev) on http://localhost:${port}`);
}
```

- [ ] **Step 4: Run the tests**

Run: `npm test`
Expected: PASS. (`new Stripe(undefined)` is never reached in the test: the empty order is refused first.)

- [ ] **Step 5: Create `.env.local` and start the server**

```bash
cp .env.example .env.local
```
Then start it with the preview tool (`preview_start name=site`), not from a shell. Open `/`; expect a blank page titled "ScottForge Reels".

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "Local dev server standing in for Vercel"
```

---

### Task 7: Brand stylesheet and the order page

**Use the `frontend-design:frontend-design` skill for the stylesheet.** The markup, ids, names and copy below are fixed, because the script and the tests depend on them. The look is authored at this step against the brand tokens in Global Constraints and the approved mockup at `C:\Users\tom\Claude\HighLightVideoMaker\.superpowers\brainstorm\1846-1791813849\content\order-page.html`. Look at scottforge.ai first and match its feel (dark coal, ember accents, Fraunces headlines with one italic ember word, mono labels).

**Files:**
- Create: `public/css/site.css`, `public/order.html`, `public/js/order.js`
- Test: `tests/pages.test.js`

**Interfaces:**
- Consumes: `price`, `TIERS` (Task 1); `validate`, `FIELDS`, `CHECKS`, `SPORTS` (Task 2); `POST /api/checkout` (Task 3).
- Produces: `site.css` classes reused by Tasks 8 and 9: `.nav`, `.wordmark`, `.btn`, `.btn-ghost`, `.label` (mono section label), `.section`, `.wrap` (centred column), `.contact-band`, `.footer`, `.prose` (text pages). Browser storage keys: `localStorage.sfrOrderDraft` (the form), `sessionStorage.sfrLastOrder` (`{ title, price, first }`, read by the thank-you page).

- [ ] **Step 1: Write the failing test** (`tests/pages.test.js`)

```js
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, existsSync } from 'node:fs';
import { FIELDS, CHECKS, SPORTS } from '../public/lib/order.js';

export const page = (name) => readFileSync(new URL(`../public/${name}.html`, import.meta.url), 'utf8');
const CAVEAT = 'You get the reel as an MP4 file and as a YouTube link. YouTube and Instagram sometimes mute or block videos that use copyrighted songs. If that happens, the MP4 is still yours to keep and share.';

test('order page has every field the server expects', () => {
  const html = page('order');
  for (const name of [...FIELDS, ...CHECKS, 'tier', 'size']) {
    assert.ok(html.includes(`name="${name}"`), `field ${name}`);
  }
  for (const sport of SPORTS) assert.ok(html.includes(`>${sport.replace('&', '&amp;')}</option>`), sport);
  for (const id of ['order', 'sumTitle', 'sumPrice', 'payBtn', 'formError']) {
    assert.ok(html.includes(`id="${id}"`), `#${id}`);
  }
});

test('order page says what the customer is agreeing to', () => {
  const html = page('order');
  assert.ok(html.includes(CAVEAT));
  for (const text of ['Ready within 5 days of your clips arriving', 'MP4 file and YouTube link',
    'One round of changes included', 'Launch price', 'tom@scottforge.ai']) {
    assert.ok(html.includes(text), text);
  }
  assert.ok(!html.includes('—'), 'no em dashes in page copy');
});

test('stylesheet defines the brand tokens', () => {
  assert.ok(existsSync(new URL('../public/css/site.css', import.meta.url)));
  const css = readFileSync(new URL('../public/css/site.css', import.meta.url), 'utf8');
  for (const token of ['#0C0B09', '#E85D2F', '#FFC857', '#F2EBDD', 'Fraunces', 'Archivo', 'JetBrains Mono']) {
    assert.ok(css.toLowerCase().includes(token.toLowerCase()), token);
  }
});
```

- [ ] **Step 2: Run to see it fail**

Run: `npm test`
Expected: FAIL, `order.html` not found.

- [ ] **Step 3: Write `public/order.html`**

Head: `<title>Order a reel | ScottForge Reels</title>`, viewport meta, the Google Fonts link for Fraunces (ital, opsz 9..144, wght 300..900), Archivo (400;500;600;700) and JetBrains Mono (400;500), `<link rel="stylesheet" href="/css/site.css">`, `<script type="module" src="/js/order.js"></script>`.

Body (structure and copy are fixed; wrapper elements and classes for layout may be added):

```html
<header class="nav">
  <a class="wordmark" href="/">Scott<em>Forge</em> Reels</a>
  <a href="/#samples">Back to samples</a>
</header>

<main class="wrap order-layout">
  <form id="order" novalidate>
    <h1>Order <em>their</em> reel.</h1>
    <p class="lede">About three minutes. You'll need a shared link to the clips.</p>

    <fieldset>
      <legend class="label">01 / Your reel</legend>
      <div class="pick">
        <label><input type="radio" name="tier" value="basic"><b>Basic</b><span>Title screen, the plays, the song</span></label>
        <label><input type="radio" name="tier" value="spotlight" checked><b>Spotlight</b><span>Animated intro, player highlighted, freeze frames, big-play effects, photos</span></label>
      </div>
      <div class="pick">
        <label><input type="radio" name="size" value="10"><b>Up to 10</b><span>plays</span></label>
        <label><input type="radio" name="size" value="20" checked><b>Up to 20</b><span>plays</span></label>
        <label><input type="radio" name="size" value="30"><b>Up to 30</b><span>plays</span></label>
      </div>
    </fieldset>

    <fieldset>
      <legend class="label">02 / The player</legend>
      <div class="grid">
        <label>First name<input name="first" autocomplete="off" maxlength="500"></label>
        <label>Last name<input name="last" autocomplete="off" maxlength="500"></label>
        <label>Jersey number<input name="number" inputmode="numeric" maxlength="500"></label>
        <label>Position<input name="position" maxlength="500"></label>
        <label>School or team<input name="school" maxlength="500"></label>
        <label>Graduation year<input name="grad" inputmode="numeric" maxlength="500"></label>
        <label>Sport<select name="sport">
          <option>Football</option><option>Basketball</option><option>Baseball</option><option>Softball</option>
          <option>Soccer</option><option>Lacrosse</option><option>Hockey</option><option>Volleyball</option>
          <option>Wrestling</option><option>Track &amp; Field</option><option>Other</option>
        </select></label>
        <label>Main team colour<input name="colour1" placeholder="Navy" maxlength="500"></label>
        <label>Second team colour<input name="colour2" placeholder="White" maxlength="500"></label>
        <label class="wide" data-spotlight>How do we spot them?
          <textarea name="spot" rows="2" maxlength="500" placeholder="White cleats, lines up wide on the left, green sleeve"></textarea></label>
      </div>
    </fieldset>

    <fieldset>
      <legend class="label">03 / The clips</legend>
      <label>Shared link (Google Drive, iCloud or Dropbox)
        <input name="clips" type="url" inputmode="url" placeholder="https://" maxlength="500"></label>
      <p class="hint">One video per play. Set the link so anyone who has it can view. Plays run in date order unless you tell us otherwise in the notes.</p>
      <p class="hint" data-spotlight>Put photos and the team logo in the same folder.</p>
    </fieldset>

    <fieldset>
      <legend class="label">04 / The song</legend>
      <label>Song and artist, or a YouTube link<input name="song" maxlength="500"></label>
      <p class="note">You get the reel as an MP4 file and as a YouTube link. YouTube and Instagram sometimes mute or block videos that use copyrighted songs. If that happens, the MP4 is still yours to keep and share.</p>
      <label class="tick"><input type="checkbox" name="song_ok">I understand the song may be muted or blocked on some platforms.</label>
    </fieldset>

    <fieldset>
      <legend class="label">05 / You</legend>
      <div class="grid">
        <label>Your name<input name="name" autocomplete="name" maxlength="500"></label>
        <label>Email (the reel is sent here)<input name="email" type="email" autocomplete="email" maxlength="254"></label>
        <label class="wide">Anything else we should know? (optional)
          <textarea name="notes" rows="3" maxlength="500"></textarea></label>
      </div>
      <label class="tick"><input type="checkbox" name="guardian">I'm this player's parent or guardian, or I have their permission to order this.</label>
      <label class="tick"><input type="checkbox" name="sample">Optional: ScottForge may show this reel on its website as a sample.</label>
    </fieldset>
  </form>

  <aside class="summary">
    <p class="label">Your order</p>
    <p id="sumTitle">Spotlight · up to 20 plays</p>
    <p id="sumPrice" class="price">$119</p>
    <p class="hint">Launch price</p>
    <ul>
      <li>Ready within 5 days of your clips arriving</li>
      <li>MP4 file and YouTube link</li>
      <li>One round of changes included</li>
    </ul>
    <p id="formError" class="error" role="alert" hidden></p>
    <button id="payBtn" class="btn" type="submit" form="order">Pay $119 →</button>
    <p class="hint">Secure card payment by Stripe</p>
    <p class="hint">Questions or something custom? <a href="mailto:tom@scottforge.ai">tom@scottforge.ai</a></p>
  </aside>
</main>

<footer class="footer">
  <a href="https://scottforge.ai">A ScottForge service</a>
  <a href="/privacy">Privacy</a> <a href="/terms">Terms</a>
</footer>
```

- [ ] **Step 4: Write `public/js/order.js`**

```js
import { price, TIERS } from '/lib/prices.js';
import { validate, CHECKS } from '/lib/order.js';

const form = document.getElementById('order');
const $ = (id) => document.getElementById(id);
const DRAFT = 'sfrOrderDraft';

function read() {
  const d = Object.fromEntries(new FormData(form));
  for (const k of CHECKS) d[k] = form.elements[k].checked;
  return d;
}

function fill(d) {
  for (const [k, v] of Object.entries(d)) {
    const el = form.elements[k];
    if (!el) continue;
    if (el.type === 'checkbox') el.checked = v === true;
    else el.value = v; // also works for a radio group
  }
}

function paint() {
  const d = read();
  const p = price(d.tier, d.size);
  $('sumTitle').textContent = `${TIERS[d.tier]} · up to ${d.size} plays`;
  $('sumPrice').textContent = `$${p}`;
  $('payBtn').textContent = `Pay $${p} →`;
  for (const el of document.querySelectorAll('[data-spotlight]')) el.hidden = d.tier !== 'spotlight';
}

function showError(field, message) {
  for (const el of form.querySelectorAll('[aria-invalid]')) el.removeAttribute('aria-invalid');
  $('formError').textContent = message;
  $('formError').hidden = false;
  const el = form.elements[field];
  const target = el instanceof RadioNodeList ? el[0] : el;
  if (target) {
    target.setAttribute('aria-invalid', 'true');
    target.focus();
  }
}

// Start from the saved draft, then let the link from the pricing table choose the package.
try {
  fill(JSON.parse(localStorage.getItem(DRAFT) ?? '{}'));
} catch { /* a damaged or blocked draft just means an empty form */ }
const params = new URLSearchParams(location.search);
if (price(params.get('tier'), params.get('size') ?? '20') !== null) form.elements.tier.value = params.get('tier');
if (price(form.elements.tier.value, params.get('size') ?? '') !== null) form.elements.size.value = params.get('size');
paint();

form.addEventListener('input', () => {
  paint();
  try {
    localStorage.setItem(DRAFT, JSON.stringify(read()));
  } catch { /* private window: the form still works, it just isn't remembered */ }
});

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const d = read();
  const v = validate(d);
  if (!v.ok) return showError(v.field, v.message);
  $('formError').hidden = true;
  $('payBtn').disabled = true;
  try {
    const res = await fetch('/api/checkout', {
      method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(d),
    });
    const out = await res.json();
    if (!res.ok) return showError(out.field, out.message);
    try {
      sessionStorage.setItem('sfrLastOrder', JSON.stringify({
        title: $('sumTitle').textContent, price: $('sumPrice').textContent, first: v.order.first,
      }));
    } catch { /* the thank-you page has a general message for this case */ }
    location.href = out.url;
  } catch {
    showError('form', "Payment couldn't start. Try again, or email us at tom@scottforge.ai.");
  } finally {
    $('payBtn').disabled = false;
  }
});
```

- [ ] **Step 5: Write `public/css/site.css`**

Invoke the `frontend-design:frontend-design` skill. Start the file with the tokens as custom properties on `:root` (`--coal`, `--coal-deep`, `--ash`, `--soot`, `--steel`, `--ember`, `--ember-hot`, `--ember-yel`, `--paper`, `--paper-dim`, `--fog`, `--fog-deep`, `--f-display: 'Fraunces', Georgia, serif`, `--f-body: 'Archivo', -apple-system, sans-serif`, `--f-mono: 'JetBrains Mono', ui-monospace, monospace`). Then style, at minimum: the shared classes listed under Produces; the order page (`.order-layout` two columns with `.summary` sticky, collapsing to one column with the summary after the form under 900 px; `.pick` radio cards whose checked state has an ember border, using `:has(input:checked)` with the radio visually hidden but focusable; `.grid` two columns with `.wide` spanning both; `.note` with an ember-yellow left rule; `.tick`; `.error` in ember-hot; `[aria-invalid]` with an ember outline; `.price` in Fraunces ember-yellow). `[hidden]` must stay hidden (`[hidden] { display: none !important; }`). Visible focus rings on every control. 16 px side gutter and no sideways scroll at 375 px.

- [ ] **Step 6: Run the tests**

Run: `npm test`
Expected: PASS.

- [ ] **Step 7: Check it in the browser**

With the `site` preview running, open `/order` and confirm each of these by reading the page or the console, not by eye alone:
1. Choosing Basic hides "How do we spot them?" and the photos hint; price shows $59 for 20 plays; the button reads "Pay $59 →".
2. All six tier and size combinations show the price from the table.
3. `/order?tier=basic&size=10` opens with Basic and 10 chosen, $39.
4. Pressing Pay on an empty form shows "First name is needed." and focuses that field.
5. Type a first name, reload: the name is still there.
6. Fill the form completely and press Pay with `.env.local` keys still empty: the message "Payment couldn't start. Try again, or email us at tom@scottforge.ai." appears and the button is usable again.
7. No console errors. Screenshot at 1440 wide and at 375 wide; no sideways scroll at 375.

- [ ] **Step 8: Commit**

```bash
git add -A && git commit -m "Order page and brand stylesheet"
```

---

### Task 8: Home page

**Use the `frontend-design:frontend-design` skill.** Layout A from the approved mockup (`...\.superpowers\brainstorm\1846-1791813849\content\home-layout.html`, left card): reel first.

**Files:**
- Create: `public/index.html` (replaces the stand-in), `public/js/home.js`, `public/js/samples.js`
- Modify: `public/css/site.css` (home sections), `tests/pages.test.js` (add tests)

**Interfaces:**
- Consumes: `site.css` shared classes (Task 7); `price` (Task 1) in the test.
- Produces: `SAMPLES` in `public/js/samples.js` (`{ spotlight: '', basic: '' }`, YouTube video ids, filled in Task 10). Until an id is set, that sample tile shows its label with "Sample coming soon" and has no play button.

- [ ] **Step 1: Add the failing tests** (append to `tests/pages.test.js`)

```js
import { price } from '../public/lib/prices.js';

test('home page prices match the price table', () => {
  const html = page('index');
  for (const tier of ['basic', 'spotlight']) {
    for (const size of [10, 20, 30]) {
      assert.ok(html.includes(`data-price="${tier}-${size}">$${price(tier, size)}<`), `${tier} ${size}`);
    }
    assert.ok(html.includes(`href="/order?tier=${tier}"`), `order link for ${tier}`);
  }
});

test('home page has every section, in order', () => {
  const html = page('index');
  const ids = ['hero', 'samples', 'pricing', 'how', 'faq', 'contact'];
  const at = ids.map((id) => html.indexOf(`id="${id}"`));
  assert.ok(at.every((i) => i > 0), `missing: ${ids.filter((_, i) => at[i] < 0)}`);
  assert.deepEqual([...at].sort((a, b) => a - b), at);
  for (const text of ['Launch price', 'tom@scottforge.ai', 'Play with sound', 'From $39',
    'data-sample="spotlight"', 'data-sample="basic"', 'href="/privacy"', 'href="/terms"']) {
    assert.ok(html.includes(text), text);
  }
  assert.ok(!html.includes('—'), 'no em dashes in page copy');
});
```

- [ ] **Step 2: Run to see them fail**

Run: `npm test`
Expected: FAIL on both new tests.

- [ ] **Step 3: Write `public/js/samples.js` and `public/js/home.js`**

`public/js/samples.js`:
```js
// YouTube video ids of the sample reels. Empty until the samples are uploaded.
export const SAMPLES = { spotlight: '', basic: '' };
```

`public/js/home.js`:
```js
import { SAMPLES } from '/js/samples.js';

// A sample loads YouTube only when it is clicked, and then plays with sound.
for (const el of document.querySelectorAll('[data-sample]')) {
  const id = SAMPLES[el.dataset.sample];
  if (!id) {
    el.classList.add('soon');
    continue;
  }
  const frame = el.dataset.frame ? document.getElementById(el.dataset.frame) : el;
  el.addEventListener('click', () => {
    const iframe = document.createElement('iframe');
    iframe.src = `https://www.youtube-nocookie.com/embed/${id}?autoplay=1&rel=0`;
    iframe.title = `${el.dataset.sample} sample reel`;
    iframe.allow = 'autoplay; encrypted-media; fullscreen';
    iframe.allowFullscreen = true;
    frame.replaceChildren(iframe);
    frame.classList.add('playing');
  }, { once: true });
}

// The hero loop is optional: without the file the hero keeps its dark background.
const loop = document.querySelector('#hero video');
loop?.addEventListener('error', () => loop.remove(), true);
```

- [ ] **Step 4: Write `public/index.html`**

Same head as the order page, with `<title>ScottForge Reels | Highlight reels for young athletes</title>`, a meta description ("Send us the clips. We cut them into a highlight reel set to their song, ready in five days. From $39."), and `<script type="module" src="/js/home.js"></script>`.

Sections, in this order, with these ids and this copy:

1. **Nav.** Wordmark "Scott*Forge* Reels" (class `wordmark`, the `em` is "Forge"). Links: Samples (`#samples`), Pricing (`#pricing`), FAQ (`#faq`). Button: "Order a reel →" to `/order`.
2. **`<section id="hero">`.** Background: `<video src="/media/hero-loop.mp4" autoplay muted loop playsinline></video>` filling the section, darkened by a gradient so text reads. Label: "Highlight reels, forged". `<h1>`: "Their best plays. *One reel.*" Lede: "Send us the clips from the stands. We cut them into a highlight reel set to their song, ready in five days." Buttons: "Order a reel →" (`/order`), and `<button class="btn-ghost" data-sample="spotlight" data-frame="heroFrame">▶ Play with sound</button>`. Small text: "From $39". An empty `<div id="heroFrame">` layered over the video, where the full reel loads.
3. **`<section id="samples">`.** Label "01 / See the difference". Heading "Same plays. *Two ways to tell it.*" Two tiles side by side, each a `<button class="sample" data-sample="basic">` / `data-sample="spotlight"` at 16:9 with a play mark and its name. Under Basic: "A title screen, the plays in order, the song." Under Spotlight: "An animated intro, your player highlighted in every play, freeze frames, big-play effects, and photos."
4. **`<section id="pricing">`.** Label "02 / Pricing". Heading "Priced by how many *plays*." Tag: "Launch price". A table with rows "Up to 10 plays", "Up to 20 plays", "Up to 30 plays" and columns Basic, Spotlight; each cell is `<td data-price="basic-10">$39</td>` and so on, written exactly in that form with the numbers from Global Constraints. Under each column an order button: `<a class="btn" href="/order?tier=basic">Order Basic →</a>` and `<a class="btn" href="/order?tier=spotlight">Order Spotlight →</a>`. One line under the table: "Every reel: ready within 5 days of your clips arriving, MP4 file and YouTube link, one round of changes included."
5. **`<section id="how">`.** Label "03 / How it works". Three numbered steps: "**Order.** Pick a package, tell us about your player and their song, and pay by card." / "**Share your clips.** Put the videos in a Google Drive, iCloud or Dropbox folder and paste the link into the order form." / "**Get your reel.** Within 5 days we email you the finished reel as a file and a YouTube link."
6. **`<section id="faq">`.** Label "04 / Questions". Each as `<details><summary>`:
   - "What footage works?" / "Phone video from the stands is fine, and 4K is best if you have it. Hudl downloads and screen recordings work too. One video per play is easiest. If all you have is full-game video, email us first."
   - "How do I share the clips?" / "Put them in a Google Drive, iCloud or Dropbox folder, set the link so anyone who has it can view, and paste it into the order form."
   - "What's the difference between Basic and Spotlight?" / "Basic is a title screen, the plays and the song. Spotlight adds an animated intro, highlights your player in every play, freezes on the big moments, and works in photos."
   - "Can I pick any song?" / the song caveat from Global Constraints, verbatim, preceded by "Yes."
   - "Which sports?" / "All of them. Tell us the sport on the order form."
   - "How long does it take?" / "Within 5 days of your clips arriving."
   - "Can I ask for changes?" / "One round of changes is included with every reel."
   - "What if I change my mind?" / "You get a full refund any time before we start work. Once the reel is delivered there are no refunds."
7. **`<section id="contact" class="contact-band">`.** "Questions or something custom? *Reach out.*" with `<a href="mailto:tom@scottforge.ai">tom@scottforge.ai</a>`.
8. **Footer.** Same as the order page.

- [ ] **Step 5: Style it** (append to `public/css/site.css`)

With the frontend-design skill: full-height hero with the video behind a coal gradient; `#heroFrame.playing` covers the hero with the iframe at 16:9; `.sample` tiles with a centred ember play mark; `.sample.soon` and `.btn-ghost.soon` show no play mark and are not clickable (`pointer-events: none`), with "Sample coming soon" on `.sample.soon` via `::after`; pricing table with the Spotlight column edged in ember; `details` with an ember marker; everything stacks to one column under 720 px; `prefers-reduced-motion` pauses nothing that matters but removes transitions.

- [ ] **Step 6: Run the tests**

Run: `npm test`
Expected: PASS.

- [ ] **Step 7: Check it in the browser**

Open `/` in the `site` preview. Confirm: sections in order; with empty sample ids both tiles read "Sample coming soon" and nothing happens on click; temporarily set `SAMPLES.spotlight` to `'M7lc1UVf-VE'`, reload, click "Play with sound", confirm an iframe from `youtube-nocookie.com` appears in the hero, then set it back to `''`; "Order Basic →" opens the order page with Basic chosen; no console errors apart from the missing `hero-loop.mp4`; screenshots at 1440 and 375 wide, no sideways scroll.

- [ ] **Step 8: Commit**

```bash
git add -A && git commit -m "Home page: reel first, samples, pricing, steps, FAQ"
```

---

### Task 9: Thank-you, privacy and terms pages

**Files:**
- Create: `public/thanks.html`, `public/privacy.html`, `public/terms.html`
- Modify: `tests/pages.test.js` (add tests), `public/css/site.css` (only if `.prose` needs more)

**Interfaces:**
- Consumes: `site.css` (`.nav`, `.wrap`, `.prose`, `.footer`, `.btn`); `sessionStorage.sfrLastOrder` and `localStorage.sfrOrderDraft` (Task 7).

- [ ] **Step 1: Add the failing tests** (append to `tests/pages.test.js`)

```js
test('thank-you page says what happens next', () => {
  const html = page('thanks');
  for (const text of ['id="bought"', 'within 5 days', 'anyone who has it', 'tom@scottforge.ai', 'sfrLastOrder', 'sfrOrderDraft']) {
    assert.ok(html.includes(text), text);
  }
});

test('privacy and terms cover what was promised', () => {
  const privacy = page('privacy');
  for (const text of ['Stripe', 'Resend', 'delete', 'sample', 'tom@scottforge.ai']) assert.ok(privacy.includes(text), `privacy: ${text}`);
  const terms = page('terms');
  for (const text of ['within 5 days', 'One round of changes', 'full refund', 'mute or block', 'permission', 'tom@scottforge.ai']) {
    assert.ok(terms.includes(text), `terms: ${text}`);
  }
  for (const name of ['thanks', 'privacy', 'terms']) {
    const html = page(name);
    assert.match(html, /<title>[^<]+ScottForge Reels<\/title>/);
    assert.ok(html.includes('/css/site.css'));
    assert.ok(!html.includes('—'), `${name}: no em dashes`);
  }
});
```

- [ ] **Step 2: Run to see them fail**

Run: `npm test`
Expected: FAIL, pages not found.

- [ ] **Step 3: Write the three pages**

All three use the order page's head (own title: "Thank you | ScottForge Reels", "Privacy | ScottForge Reels", "Terms | ScottForge Reels"), nav and footer, with the content in `<main class="wrap prose">`.

`thanks.html` content:
```html
<h1>Order <em>received.</em></h1>
<p id="bought">Thanks for your order.</p>
<p>A confirmation email is on its way. Here's what happens next:</p>
<ol>
  <li>Check that your clips link opens for anyone who has it. That's the one thing that holds reels up.</li>
  <li>We build the reel. It will be ready within 5 days of your clips arriving.</li>
  <li>You get an MP4 file and a YouTube link by email. One round of changes is included.</li>
</ol>
<p>Questions? <a href="mailto:tom@scottforge.ai">tom@scottforge.ai</a></p>
<p><a class="btn" href="/">Back to the home page</a></p>
<script>
  try {
    const o = JSON.parse(sessionStorage.getItem('sfrLastOrder') || 'null');
    if (o) document.getElementById('bought').textContent =
      'Thanks. We have your order for ' + o.first + "'s reel: " + o.title + ' (' + o.price + ').';
    localStorage.removeItem('sfrOrderDraft');
  } catch (e) { /* storage blocked: the general message stays */ }
</script>
```

`privacy.html` content:
```html
<h1>Privacy</h1>
<p class="hint">Last updated October 2026</p>
<p>ScottForge Reels is run by ScottForge in Youngstown, Ohio. This page says what we collect when you order a reel and what we do with it.</p>
<h2>What we collect</h2>
<p>What you type into the order form: the player's name, jersey number, position, school or team, graduation year, sport and team colours; your name and email address; the link to your clips; your song choice; and any notes. We also receive the videos and photos you share through that link.</p>
<h2>What we use it for</h2>
<p>To make and deliver the reel you ordered, and to write to you about that order. Nothing else. We don't sell or share your details, and we don't send marketing email.</p>
<h2>Payment</h2>
<p>Card payments are taken by Stripe. Your card details go straight to Stripe and never reach us. Stripe keeps a record of the order, including the form details above.</p>
<h2>Email</h2>
<p>Order emails are sent through Resend, an email delivery service.</p>
<h2>The finished reel</h2>
<p>We send you the reel as a file and upload it to YouTube as an unlisted video, which means only people with the link can watch it. We show a reel on our website as a sample only if you ticked that box on the order form. You can change your mind at any time.</p>
<h2>Children</h2>
<p>Reels usually feature players under 18. We only take orders from a parent or guardian, or someone with their permission, and the order form asks you to confirm that.</p>
<h2>Deleting your information</h2>
<p>Email <a href="mailto:tom@scottforge.ai">tom@scottforge.ai</a> and we will delete your footage, your reel, the YouTube upload and your order details, apart from the payment record we are required to keep.</p>
```

`terms.html` content:
```html
<h1>Terms</h1>
<p class="hint">Last updated October 2026</p>
<h2>What you get</h2>
<p>A highlight reel made from the clips you share, in the package you chose (Basic or Spotlight, up to 10, 20 or 30 plays), delivered as an MP4 file and an unlisted YouTube link.</p>
<h2>When you get it</h2>
<p>Your reel will be ready within 5 days of your clips arriving. If the link doesn't open or the clips are missing, the 5 days start when we can reach them.</p>
<h2>Changes</h2>
<p>One round of changes is included. Send them in one email after you've watched the reel.</p>
<h2>Refunds</h2>
<p>You get a full refund any time before we start work on your reel. Once the reel has been delivered there are no refunds.</p>
<h2>Your footage and your song</h2>
<p>By ordering you confirm that you are the player's parent or guardian or have their permission, and that you have the right to share the videos and photos you send us.</p>
<p>You choose the song. YouTube, Instagram and other platforms sometimes mute or block videos that use copyrighted songs. If that happens, the MP4 is still yours to keep and share.</p>
<h2>Questions</h2>
<p><a href="mailto:tom@scottforge.ai">tom@scottforge.ai</a></p>
```

- [ ] **Step 4: Run the tests**

Run: `npm test`
Expected: PASS.

- [ ] **Step 5: Check in the browser**

Open `/thanks`, `/privacy`, `/terms`: each renders in the brand style with nav and footer, readable line length, no console errors. In the console on `/thanks` run `sessionStorage.setItem('sfrLastOrder', JSON.stringify({ title: 'Spotlight · up to 20 plays', price: '$119', first: 'Caleb' }))`, reload, and confirm the line reads "Thanks. We have your order for Caleb's reel: Spotlight · up to 20 plays ($119)."

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "Thank-you, privacy and terms pages"
```

---

### Task 10: Samples, deployment and a real test order

This task needs Tom at several points. Ask for each item when it's reached (in the session and on Telegram), and carry on with whatever doesn't depend on it. Never create accounts, enter real card details, or handle live keys on Tom's behalf; he types keys into `.env.local` and Vercel himself.

**Files:**
- Modify: `public/js/samples.js`
- Create: `public/media/hero-loop.mp4`, `README.md`

- [ ] **Step 1: Sample reels (uses the Highlight Reel Maker app, no code changes there)**

Read `C:\Users\tom\Claude\HighLightVideoMaker\STATUS.md` first. On a scratch copy of the Caleb project (never the real one), pick about 10 plays that have spotlight clicks and hits, and build two wide reels from the same plays: Spotlight (animated intro, ring, freeze, beat sync, one or two big plays) and Basic (classic intro, highlight off, freeze off, no photos, no big plays). Don't restart Tom's app while a build is running. Show Tom both files and get his OK on them.

- [ ] **Step 2: Tom uploads both to YouTube as Unlisted** and gives the two video ids (or approves uploading them through the app's Publish screen). Put them in `public/js/samples.js`.

- [ ] **Step 3: Cut the hero loop**

```bash
ffmpeg -ss <start of the best 10 seconds> -t 10 -i "<spotlight reel>.mp4" -an -vf "scale=1280:-2,fps=30" -c:v libx264 -crf 28 -preset slow -movflags +faststart public/media/hero-loop.mp4
```
Choose the start so the loop holds the intro's end and a highlighted play. Expected: a file under 4 MB. Check it plays in the hero.

- [ ] **Step 4: Write `README.md`**

Cover, briefly: what the site is; `npm test`; `npm run dev`; the six settings from `.env.example` and where each comes from; how to change a price (`public/lib/prices.js`, then the matching cell in `public/index.html`; the test fails if they disagree); how to add a sample (`public/js/samples.js`); where orders are recorded (Stripe dashboard, Payments, the payment's metadata).

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "Sample reels, hero loop, README"
```

- [ ] **Step 6: Accounts (Tom)**

Tom does each of these himself:
1. Buys `forgedreels.com` and `forgedreel.com`.
2. Opens a Stripe account and copies the **test** secret key (`sk_test_...`) into `.env.local`.
3. Opens a Resend account, adds the domain `forgedreels.com`, adds the DNS records Resend shows, and puts the API key in `.env.local`.
4. Authorises the Vercel connector (`/mcp` in a Claude Code terminal) or runs `npx vercel login`.

- [ ] **Step 7: Deploy a preview**

Create a private GitHub repository `ScottForgeReels` (ask Tom before pushing), push `main`, import it into Vercel as a project with the "Other" preset, and add the six settings for the Preview and Production environments (test-mode Stripe values for now; `SITE_URL` is the deployment's own address). Deploy. Expected: `/`, `/order`, `/thanks`, `/privacy`, `/terms` all load on the preview address.

- [ ] **Step 8: Connect the webhook (Tom, in Stripe test mode)**

In Stripe: Developers, Webhooks, Add endpoint: `<preview address>/api/stripe-webhook`, event `checkout.session.completed`. Copy the signing secret into Vercel as `STRIPE_WEBHOOK_SECRET` and redeploy.

- [ ] **Step 9: Test order end to end**

On the preview address, Tom places a Spotlight, up to 20 plays order using Stripe's test card `4242 4242 4242 4242`, any future date, any CVC. Confirm all of:
1. Stripe's page shows "Spotlight highlight reel, up to 20 plays" at $119.00.
2. After paying, the thank-you page names the player and package.
3. Tom's inbox has "New reel order: ..." with every field; replying addresses the customer.
4. The customer address has the confirmation; replying addresses tom@scottforge.ai.
5. In Stripe, the payment shows the order fields under metadata.
6. Go back to `/order`, press Pay, then use Stripe's back link: the form is still filled in.

If an email is missing, read the function's log in Vercel and the webhook attempt in Stripe before changing anything.

- [ ] **Step 10: Go live (Tom's decision)**

Point `forgedreels.com` at the Vercel project and redirect `forgedreel.com` to it. Switch the Production settings to Stripe's live secret key and a live-mode webhook endpoint with its own signing secret; set `SITE_URL` to `https://forgedreels.com`. Tom places one real Basic, up to 10 plays order ($39) with his own card, checks both emails, then refunds it in Stripe.

- [ ] **Step 11: Follow-ups to record**

Tell Tom: `https://forgedreels.com/privacy` now exists, which is what the YouTube API review for the reel maker was waiting on. Add a line to `C:\Users\tom\Claude\HighLightVideoMaker\STATUS.md` saying the sales site exists, where its repository is, and that recruiting packages were deliberately left for later.
