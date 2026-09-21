"""
demo_load_side.py  (Person A)  --  live demo of the "Predict Load" stage.

Simulates the 60 s control loop over three services (one of each pattern):
every tick it slides a window over each service's load trace, classifies the
pattern, routes to ARIMA / LSTM / XGBoost and prints the forecast L that
Person C's fusion engine will turn into a replica count K.

    python demo_load_side.py                 # 4 ticks, 1.5 s pause between ticks
    python demo_load_side.py --ticks 8 --delay 0
    python demo_load_side.py --classifier tree
"""
import argparse
import time
import warnings

warnings.filterwarnings("ignore")

from data_gen.synthetic_data import generate_service_traces  # noqa: E402
from models.load_pipeline import classify_and_forecast, format_decision, warm_up  # noqa: E402

WINDOW = 150


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=4)
    ap.add_argument("--delay", type=float, default=1.5, help="seconds between ticks")
    ap.add_argument("--classifier", choices=["rules", "tree"], default="rules")
    args = ap.parse_args()

    print("FAP-Scale | Stage 2: workload classification + dynamic model selection")
    print("loading models (cached LSTM)...", end=" ", flush=True)
    t0 = time.time()
    warm_up(args.classifier)
    print(f"ready in {time.time() - t0:.1f}s\n")

    traces = generate_service_traces(n_services=3, length=WINDOW + args.ticks + 1)
    for tick in range(args.ticks):
        now = WINDOW + tick
        print(f"── tick {tick + 1}  (t = {now} min) " + "─" * 60)
        for sid, tr in traces.items():
            out = classify_and_forecast(tr["load"][now - WINDOW:now], classifier=args.classifier)
            actual = tr["load"][now]
            err = abs(out["predicted_load"] - actual) / actual
            print(f"  {sid} [true {tr['pattern']:<8s}] {format_decision(out)} "
                  f"| actual={actual:6.2f} err={err:5.1%}")
        print("  -> L values handed to fusion engine (Person C)\n")
        time.sleep(args.delay)


if __name__ == "__main__":
    main()
