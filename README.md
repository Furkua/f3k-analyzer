# f3k-analyzer
F3K Session Analyzer is for digging into discus launch glider (DLG) flight logs.
Drop in a bunch of your CSVs (or even a whole ZIP of them) and it’ll:

	•	Line up all your sessions so the first launch starts at the same time
	•	Spot each throw and mark the peak altitude
	•	Show you an interactive graph you can zoom, pan, and poke at
	•	Give you tables with max heights, durations, climb rates, and more
	•	Summarize each session so you can see your best throws and overall trends
	•	Let you download all the throw data with your own notes and tags

It’s basically a flight diary and scoreboard rolled into one, perfect for comparing training days, trying different setups, or just geeking out over numbers.
try it here: http://furkua.streamlit.app

- New predictive features:
  • Throw Strength Prediction
  • Session Fatigue Modeling
  • Best Possible Throw Estimate
  • Optimal Launch Timing in Tasks (highlighted in plot)

- Layout improvements:
  • Added "Classic" vs "Wide + X-Zoom" view toggle
  • Wide view has large top-centered plot sized like design mockup
  • Horizontal scroll zoom with Y-axis lock, double-click to reset, range slider

- General improvements:
  • Better plot alignment and sizing
  • Peak markers + hover tooltips preserved in both modes
