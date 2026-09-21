# Deploying the signal bot

The bot runs one cycle per invocation: **fetch → retrain → predict → decide →
notify → exit**. It is stateless, so any scheduler works.

> **Read this first.** The research behind this bot found **no directional edge**
> in XAUUSD at 1m/5m/15m. Across seven studies, a *perfect* direction oracle still
> lost money because the median 1-minute move (1.69 bp) is smaller than the
> round-trip cost (1.70 bp). So the bot will send `NO TRADE` almost every time,
> and that is the correct output — not a malfunction, and not something to tune
> away. Nothing here is financial advice.

## Required credentials

Two secrets, set in whichever platform you deploy to. Never commit them.

| Variable | Value |
|---|---|
| `TELEGRAM_BOT_TOKEN` | from @BotFather |
| `TELEGRAM_CHAT_ID` | the chat to post to |

Get your chat id by messaging the bot, then:

```bash
curl -s "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/getUpdates" \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['result'][0]['message']['chat']['id'])"
```

## Option A — GitHub Actions (no server)

The workflow is at `.github/workflows/signal-bot.yml`. It runs every 15 min during
London/NY gold hours (07:00–21:00 UTC, Mon–Fri) plus a daily heartbeat.

**One manual step is required.** Add the two secrets, then enable the workflow:

1. Repo → **Settings → Secrets and variables → Actions → New repository secret**
2. Add `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`
3. Repo → **Actions** tab → enable workflows if prompted
4. **Actions → XAUUSD signal bot → Run workflow** (tick *dry run* first)

> This step could not be automated: the available token has no `actions:secrets`
> permission (`Resource not accessible by integration`), and writing credentials
> into the repo instead would leak them.

**Optional repository variables** (Settings → Variables) to change behaviour:
`SIGNAL_SYMBOL`, `SIGNAL_BAR`, `SIGNAL_HISTORY`, `SIGNAL_PAYOUT`.

**Limitations to know:** GitHub cron is best-effort and can be delayed minutes
under load, and scheduled workflows are disabled after 60 days of repo
inactivity. Use Option B if timing matters.

## Option B — Cloud Run / any container (recommended for timing)

Same code, more reliable scheduling.

```bash
# build and push
gcloud builds submit --tag gcr.io/$PROJECT/xauusd-bot

# store secrets in Secret Manager, not in the image
echo -n "$TELEGRAM_BOT_TOKEN" | gcloud secrets create telegram-bot-token --data-file=-
echo -n "$TELEGRAM_CHAT_ID"   | gcloud secrets create telegram-chat-id   --data-file=-

# deploy as a job with a schedule (fires at bar boundaries)
gcloud run jobs create xauusd-bot \
  --image gcr.io/$PROJECT/xauusd-bot \
  --command python --args "-m,src.cloud_run" \
  --set-secrets TELEGRAM_BOT_TOKEN=telegram-bot-token:latest,TELEGRAM_CHAT_ID=telegram-chat-id:latest \
  --max-retries 1 --task-timeout 300

gcloud scheduler jobs create http xauusd-bot-15m \
  --schedule "*/15 7-21 * * 1-5" --uri "$(gcloud run jobs describe xauusd-bot --format='value(...)')" \
  --time-zone UTC
```

Or run it as an always-on service with the internal scheduler, which keeps its own
clock, skips closed markets, and retries with backoff:

```bash
gcloud run deploy xauusd-bot --image gcr.io/$PROJECT/xauusd-bot \
  --command python --args "-m,src.scheduler" \
  --set-secrets TELEGRAM_BOT_TOKEN=telegram-bot-token:latest,TELEGRAM_CHAT_ID=telegram-chat-id:latest \
  --min-instances 1 --no-cpu-throttling
```

## Option C — a VM with cron

```bash
git clone <repo> && cd Cyto-bot
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
crontab -e
```

```cron
*/15 7-21 * * 1-5  cd /opt/Cyto-bot && .venv/bin/python -m src.cloud_run 2>&1 | logger -t xauusd-bot
```

## Local testing

```bash
pip install -r requirements.txt

python -m src.cloud_run --dry-run              # print the message, send nothing
python -m src.cloud_run                        # send for real
python -m src.cloud_run --heartbeat --dry-run  # include the alive header
python -m src.scheduler --once                 # one cycle via the scheduler
python -m src.scheduler                        # loop, bar-aligned

pytest tests/ -q                               # 87 tests
```

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `SIGNAL_SYMBOL` | `GC=F` | Yahoo symbol (COMEX gold front month) |
| `SIGNAL_BAR` | `5min` | bar size: `1min`, `5min`, `15min` |
| `SIGNAL_HISTORY` | `5d` | fetch window; needs ≥500 bars after resampling |
| `SIGNAL_PAYOUT` | `0.80` | binary payout; sets the breakeven bar |
| `SIGNAL_HORIZON` | `1` | bars ahead to predict |
| `SIGNAL_SETTLE_SECONDS` | `20` | delay after a bar close before fetching |
| `SIGNAL_INTERVAL_SECONDS` | `0` | `0` = align to bars; else fixed interval |

Raising `SIGNAL_PAYOUT` lowers the breakeven (`1/(1+payout)`) and therefore makes
the bot *more* willing to trade. That is arithmetic, not an improvement in skill —
do not use it to manufacture signals.

## What the message means

```
NO TRADE — no edge over the payout bar
P(next bar up): 0.4559
Breakeven at 80% payout: 0.5556
Edge over breakeven: -0.0114
```

`Edge` is the model's best side minus the breakeven rate. It trades only when this
clears `0` plus a 0.005 margin. Every message carries the evidence caveat,
including on the rare occasion it says `UP` or `DOWN`.

## Failure handling

- A market-data failure sends one error message and exits non-zero, so the
  scheduler flags it rather than silently going quiet.
- GitHub Actions sends a separate failure notice with a link to the run.
- The scheduler retries with exponential backoff (30 s → 600 s) and stops cleanly
  on `SIGTERM`/`SIGINT`.
- Each cycle appends to `results/signals.jsonl`, uploaded as a CI artifact.

## What I would not do

Do not lower `SIGNAL_PAYOUT`/`min_edge` or add model complexity hoping the bot
starts emitting directional signals. That would only add confidence to noise. The
measured result is that no directional model clears costs here; the honest value
of this bot is that it *declines* correctly and tells you when it cannot know.