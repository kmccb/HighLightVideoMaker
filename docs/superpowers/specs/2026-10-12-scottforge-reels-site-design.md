# ScottForge Reels: sales website design

_Approved in conversation 2026-10-12. This is the spec for version one._

## Purpose

Sell done-for-you highlight reels to parents of young athletes. A parent watches a sample, picks a package, fills in one form, pays by card and shares a link to their clips. Tom builds the reel in the Highlight Reel Maker app on his PC and sends it back.

Success for version one: a stranger can go from landing on the site to a paid order, with everything Tom needs to make the reel arriving in one email, without Tom touching the site.

## Decisions

| Topic | Decision |
|---|---|
| What is sold | Done-for-you reels only. The app itself is not sold. |
| Name | ScottForge Reels |
| Domain | `forgedreels.com`, with `forgedreel.com` redirecting to it |
| Branding | Same tokens, fonts and voice as scottforge.ai |
| Sports | All sports |
| Turnaround | Within 5 days of the clips arriving |
| Revisions | One round included |
| Refunds | Full refund any time before work starts, none after delivery |
| Music | Customer's own song. Caveat about muting and blocking shown on the order form, with a required tick box. |
| Delivery | MP4 file plus an unlisted YouTube link on the ScottForge channel |
| Samples | Caleb only at launch: one Basic and one Spotlight reel made from the same plays |
| Contact | `tom@scottforge.ai` |

## Packages and prices

Launch prices, in US dollars.

| Plays | Basic | Spotlight |
|---|---|---|
| Up to 10 | 39 | 79 |
| Up to 20 | 59 | 119 |
| Up to 30 | 79 | 159 |

- **Basic:** a standard title screen, the plays, the song.
- **Spotlight:** animated intro, the player highlighted in each play, freeze frames, big-play effects, photos.

The prices live in one file (`lib/prices.js`). The order page reads it to show the price; the checkout function reads it to charge. The amount charged is always computed on the server from tier and size.

## Out of scope for version one

Recruiting packages, customer accounts, uploading clips to the site, order tracking, discount codes, team rates, add-ons. The page structure must not block adding recruiting as a third tier later.

## Pages

### Home (`/`)

In this order:

1. **Nav.** ScottForge Reels wordmark, links to Samples, Pricing, FAQ, and an "Order a reel" button.
2. **Hero.** Caleb's Spotlight reel full-width. A short loop plays muted on load (self-hosted MP4, about 10 seconds, a few MB). A "Play with sound" button opens the full reel from YouTube with audio. Headline, one line of supporting text, "Order a reel" button, "From $39".
3. **See the difference.** Basic and Spotlight samples side by side, same plays, each a YouTube embed that loads on click.
4. **Pricing.** The grid above, with what each tier includes, a "Launch price" label, and an order button per tier that opens the order page with that tier chosen.
5. **How it works.** Three steps: order, share your clips, get your reel in 5 days.
6. **FAQ.** What footage works, how to share clips, the song caveat, turnaround, revisions, refunds, which sports.
7. **Contact band.** "Questions or something custom? Reach out." with the email address.
8. **Footer.** Link to scottforge.ai, privacy, terms.

### Order (`/order`)

One page, five numbered sections, and a summary box that stays in view with the chosen package, live price, the three promises (5 days, MP4 and YouTube link, one round of changes) and the pay button.

| Section | Fields |
|---|---|
| 01 Your reel | Tier (Basic, Spotlight). Size (10, 20, 30). Both can be preset from the URL (`?tier=spotlight&size=20`). |
| 02 The player | First name, last name, jersey number, position, school or team, graduation year, sport, two team colours. Spotlight only: "How do we spot them?" free text. |
| 03 The clips | One shared link (Google Drive, iCloud or Dropbox). Helper text: one video per play; photos and the team logo go in the same folder (Spotlight); plays run in date order unless the notes say otherwise. |
| 04 The song | Song and artist, or a YouTube link. The caveat text. Required tick box: "I understand the song may be muted or blocked on some platforms." |
| 05 You | Name, email, optional notes. Required tick box: parent, guardian or has permission. Optional tick box: ScottForge may show this reel as a sample. |

Required: tier, size, first and last name, sport, clips link, song, customer name, email, and the two required tick boxes. Jersey number is required for Spotlight. Free-text fields are capped at 500 characters, which is Stripe's limit per stored value.

The contact line appears near the pay button.

The form saves its fields in the browser as the parent types, so a parent who backs out of payment returns to a filled-in form.

### Thank you (`/thanks`)

Confirms the order, repeats what was bought, and says what happens next: a confirmation email is on its way, make sure the clips link is shared with anyone who has it, the reel arrives within 5 days.

### Privacy (`/privacy`) and Terms (`/terms`)

Short, plain-language pages. Privacy: what is collected (the form fields), why, that payment is handled by Stripe, that footage is used only to make the reel unless the sample box was ticked, how to ask for deletion. Terms: what is delivered, turnaround, the revision, the refund rule, the customer's confirmation that they may share the footage, the song caveat.

The privacy page is also what the YouTube API review for the reel maker app needs.

## Architecture

A static site on Vercel with two serverless functions. No database.

```
site/
  index.html  order.html  thanks.html  privacy.html  terms.html
  css/site.css            brand tokens copied from scottforge.ai, then page styles
  js/order.js             form behaviour: live price, Spotlight-only fields, saving, validation
  lib/prices.js           the price table and a price(tier, size) lookup
  lib/order.js            field list, validation and length caps, shared by browser and server
  api/checkout.js         validates an order, creates a Stripe Checkout Session
  api/stripe-webhook.js   verifies Stripe's signature, sends the two emails
  lib/email.js            builds and sends the order and confirmation emails
  media/hero-loop.mp4
```

The site lives in its own folder and git repository (`C:\Users\tom\Claude\ScottForgeReels`), separate from the reel maker.

### Order flow

1. The browser posts the form as JSON to `/api/checkout`.
2. `checkout` validates every field with `lib/order.js`, looks up the price with `lib/prices.js`, and creates a Stripe Checkout Session: one line item named for the package, the customer's email prefilled, every order field stored in the session's metadata, success URL `/thanks`, cancel URL `/order`.
3. The browser goes to Stripe's hosted payment page.
4. On payment, Stripe calls `/api/stripe-webhook` with `checkout.session.completed`.
5. The webhook checks the signature, reads the order from the metadata, and sends two emails: the full order to Tom, and a confirmation to the customer with a reply-to of Tom's address.

Stripe is the record of orders. Every paid order and its fields can be read in the Stripe dashboard.

### Services and settings

| Service | Used for | Setting |
|---|---|---|
| Stripe | Checkout and order record | `STRIPE_SECRET_KEY`, `STRIPE_WEBHOOK_SECRET` |
| Resend | Sending the two emails | `RESEND_API_KEY` |
| Vercel | Hosting and functions | `ORDER_EMAIL_TO` (Tom's address), `SITE_URL` |

Tom creates the Stripe, Resend and domain accounts himself and authorises Vercel.

## Error handling

| Case | Behaviour |
|---|---|
| Missing or invalid field | The form marks the field and does not submit. The server repeats the same checks and answers 400 with the field name. |
| Tier or size not in the price table | Server answers 400. No session is created. |
| Stripe cannot create a session | The form shows "Payment couldn't start. Try again, or email us." with the contact address. |
| Customer cancels on Stripe | Back to `/order` with the form still filled in. |
| Webhook signature fails | 400, nothing is sent. |
| Email sending fails | The webhook answers 500 so Stripe retries. The order remains in Stripe either way. |
| Stripe delivers the same event twice | The customer may get two confirmations. Accepted for version one. |

## Brand

Tokens from scottforge.ai: coal `#0C0B09`, ash `#141210`, soot `#1C1915`, steel `#2A2520`, ember `#E85D2F`, ember-hot `#FF8A3D`, ember-yellow `#FFC857`, paper `#F2EBDD`, paper-dim `#D4CBB9`, fog `#8A8276`. Fraunces for display (italic ember for the emphasised word), Archivo for body, JetBrains Mono for small labels. Numbered section labels ("01 / Pricing"). Plain, first-person voice.

## Testing

- Unit tests for `price(tier, size)` over all six combinations and for rejected inputs.
- Unit tests for order validation: each required field, Spotlight-only rules, length caps.
- Function tests with Stripe and Resend mocked: checkout builds the right session; the webhook rejects a bad signature and sends both emails for a good event.
- One end-to-end run in Stripe test mode with a test card: form, payment, thank-you page, both emails received.
- Layout checked at phone and desktop widths.

## Before launch

- Tom buys the domains and creates the Stripe and Resend accounts.
- Build Caleb's Basic and Spotlight sample reels from the same plays, upload both unlisted, cut the hero loop.
- Switch Stripe from test to live keys and place one real order.
