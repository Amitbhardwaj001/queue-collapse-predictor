# Expo Guide

## Before the expo

### 1. Test on the expo laptop
- Double-click **`start.bat`**; the first run trains the four domains (a few minutes, once). Then the dashboard
  opens at http://127.0.0.1:5000.
- Everything works **without internet** (YOLO weights and Chart.js are stored in the project).
- Allow the browser to use the webcam once, so it doesn't ask in front of the judges.

### 2. Record your own queue footage (makes the camera demo convincing)
- **Where:** college canteen at lunch time (12:30–1:30), when the queue grows.
- **Camera position:** as **high** as possible, looking **down and along** the queue.
- **Keep the phone still.** 3–10 minutes of landscape .mp4. The system samples one frame every 10 s.
- Ask canteen staff for permission before filming, and avoid close-ups of faces.

Then in the dashboard: *Physical queues → Main Canteen* → upload the video → draw the **Queue area** → **Save area**.

### 3. Print the poster
Open http://127.0.0.1:5000/poster → write your team names in `queue_predictor/templates/poster.html`
→ **Ctrl+P** → Save as PDF (A3, portrait, "Background graphics" ticked) → print.

## 4-minute demo script

1. **Problem (20 s).** "Canteens, hospitals, web servers and call centres all collapse the same way: demand
   briefly beats capacity, a backlog builds and waiting explodes. Monitoring tells you the backlog *now*. We tell you
   *when it will collapse* and *how many servers stop it*."
2. **One engine, four domains (20 s).** Click the four tabs at the top. "Same engine; only the vocabulary,
   thresholds and demand change."
3. **Collapse Radar (80 s).** *Physical queues → Demo Canteen → Start.* As the lunch rush builds the radar goes
   SAFE → AT RISK → COLLAPSE IMMINENT. Point at **time to collapse** ("about 6 minutes, 80 % range 3–10"), the
   **probability**, and the **purple uncertainty band** on the chart.
4. **What-if staffing (50 s).** In the *What-if staffing simulator* table show that 2 counters means an 88 % chance
   of collapse and 4 counters means about 2 %. Press **Apply** on the safe row: the forecast and risk drop.
   "This is the decision the manager makes *before* the queue collapses."
5. **Stress test (30 s).** Switch to *Web / API servers → Start*, then press **Inject surge** ("flash sale") and
   watch the radar react within a few ticks. Do the same on the hospital or call centre if time allows.
6. **Camera (30 s).** Back to *Physical queues → Main Canteen*: upload your footage or start the webcam with 3–4 visitors
   in a line. Point at the green boxes and the queue area.
7. **Proof (30 s).** Scroll to *Early-warning study*: reactive alarm 0 minutes ahead versus the Collapse Radar's
   mean warning time and the false alarms per day. Say openly that sudden bursts cannot be predicted.

## Questions judges are likely to ask

**Where did the training data come from?**
Simulated days per domain (rush hours, bursts, weekends, under-staffing, people giving up), because no public
minute-by-minute backlog datasets exist. Every count the system sees is saved, so it can be retrained on real data.
*Say this openly; don't claim it is real data.*

**What is new compared with a normal people counter or a monitoring dashboard?**
They show the present. This system predicts the *future*, gives a *probability and a time* of collapse, and
simulates *staffing options* to recommend an action. It also works for any system with arrivals, capacity and a backlog.

**What is the Collapse Radar?**
It converts the XGBoost forecast into an arrival rate per minute and replays it 500 times with random arrivals and
service (a Monte Carlo queue simulation) for each staffing option. The fraction of runs where the wait reaches the
critical level is the collapse probability; the first crossing time is the time to collapse.

**How do you know it helps?**
The early-warning study on unseen days compares a reactive alarm (fires at the critical wait, 0 minutes ahead) with the
classifier and the radar: how many collapses were warned about early, the mean warning time and false alarms per day.

**Why XGBoost and not an LSTM?** The inputs are a small table of features. Gradient-boosted trees are the strongest
method for that kind of data, train in a minute without a GPU and are easy to explain.

**Why YOLOv8?** Fast enough for a laptop CPU, pretrained on COCO where "person" is a class.

**How is waiting time calculated?** Backlog ÷ (open servers × items one server handles per minute).

**How would you add a new domain?** One `Domain(...)` entry in `domains.py` (vocabulary, thresholds, demand), then
`python -m queue_predictor.train --domain <key>`.

**What are the limitations?** Trained on simulated data; only physical queues sense their backlog automatically;
sudden bursts cannot be predicted in advance. Next steps: real-data connectors, ByteTrack person tracking, alerts.
