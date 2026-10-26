# Expo Guide

## Before the expo

### 1. Record your own queue footage (most important)
Real footage from your own college convinces judges far more than anything else.

- **Where:** college canteen at lunch time (12:30–1:30), when the queue grows.
- **Camera position:** as **high** as possible (first-floor railing, top of a shelf, held above
  your head), looking **down and along** the queue. People hiding behind each other is the
  main cause of missed counts, and a high angle fixes that.
- **Keep the phone still.** Rest it on something; don't walk around.
- **Length:** 3–10 minutes. The system samples one frame every 10 s, so a 5-minute clip
  gives 30 readings, enough to show a real growing or shrinking trend.
- **Format:** normal phone video (.mp4) is fine. Landscape is better.
- Take 2–3 still photos as well (quiet queue, medium queue, long queue).
- Ask canteen staff for permission before filming, and avoid close-ups of faces.

Then in the dashboard: *Main Canteen* → upload the video → draw the **Queue area** around
where people actually line up → **Save area** → upload the video again for clean counts.

### 2. Test on the expo laptop
- Double-click **`start.bat`**; the dashboard opens at http://127.0.0.1:5000.
- Everything works **without internet** (YOLO weights and Chart.js are stored in the project).
- Allow the browser to use the webcam once, so it doesn't ask in front of the judges.

### 3. Print the poster
Open http://127.0.0.1:5000/poster → write your team names in `queue_predictor/templates/poster.html`
→ **Ctrl+P** → Save as PDF (A3, portrait, "Background graphics" ticked) → print.

## 3-minute demo script

1. **Problem (20 s).** "Queues are managed reactively: by the time staff notice, people are
   already waiting 15 minutes. CCTV shows the queue *now*; we predict it 10–15 minutes *ahead*."
2. **Detection (40 s).** Upload your canteen photo/video. Point at the green boxes and the count.
   Show the queue area: people outside it are greyed out and not counted.
3. **Live webcam (30 s).** Start the camera and ask 3–4 visitors to stand in a line. The count updates.
4. **Prediction (60 s).** Switch to *Demo Canteen*, press **Start**. As the lunch rush builds,
   the risk goes LOW → MEDIUM → HIGH and the system says *"Open 2 more counters now"*.
   Then change **Open counters** from 2 to 4 and press Save: the predicted wait drops and the
   risk falls. "That is the decision the manager can make *before* the queue collapses."
5. **Results (30 s).** Point at the Model accuracy panel / poster: AUC 0.974 for detecting
   critical congestion; at rush hour the 10-minute forecast is about 33 % more accurate
   than assuming the queue stays the same.

## Questions judges are likely to ask

**Where did the training data come from?**
We simulated 150 days of canteen queues (rush hours, class-break surges, weekends,
understaffing, people leaving long queues), because no public minute-by-minute queue dataset
exists. Every count the system detects is saved, so after a few weeks of real camera data
the same pipeline retrains on real data. *Say this openly; don't claim it is real data.*

**How accurate is it?**
Critical-congestion detection: AUC 0.974. 10-minute forecast: R² 0.88; during busy periods
the average error is ~5.5 people versus 8.2 for the naive guess. Tested on later days the
model never saw during training.

**Why YOLOv8?** It's fast enough to run on a normal laptop CPU, pretrained on the COCO dataset
where "person" is one of the classes, and accurate for people.

**Why XGBoost and not deep learning (LSTM)?** The inputs are a small set of table-style features
(recent queue lengths, time of day, counters). Gradient-boosted trees are the strongest
method for that kind of data, train in about a minute without a GPU, and are easy to explain.

**How is waiting time calculated?** People in queue ÷ (open counters × people served per
counter per minute). The service time per person is set per location.

**What if people block each other?** Mount the camera high and use the queue area. A future
step is person tracking (ByteTrack) to follow each person through the queue.

**What is new compared with a normal people counter?** A people counter tells you the present.
This system predicts the future, gives a probability of collapse, and recommends an action.
