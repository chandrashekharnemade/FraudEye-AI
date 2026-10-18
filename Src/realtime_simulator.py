
from pathlib import Path
import argparse
import csv
import time
from datetime import datetime, timezone

import pandas as pd
import requests
from time import perf_counter

# Project paths
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "raw" / "creditcard.csv"
OUTPUT_PATH = ROOT / "results" / "live_predictions.csv"

API_URL = "http://127.0.0.1:8000/predict"
HEALTH_URL = "http://127.0.0.1:8000/health"


def main():
    parser = argparse.ArgumentParser(
        description="FraudEye AI API-connected transaction simulator"
    )
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument(
    "--continuous",
    action="store_true",
    help="Continue generating transactions until Ctrl+C"
)
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument(
        "--mode",
        choices=["demo", "realistic"],
        default="demo",
        help="demo mixes classes; realistic samples the dataset naturally"
    )
    args = parser.parse_args()

    if args.limit < 1:
        raise SystemExit("Error: --limit must be at least 1.")

    if args.delay < 0:
        raise SystemExit("Error: --delay cannot be negative.")

    if not DATA_PATH.exists():
        raise SystemExit(f"Dataset missing: {DATA_PATH}")

    # Check API availability
    try:
        health_response = requests.get(HEALTH_URL, timeout=5)
        health_response.raise_for_status()
        print("API health:", health_response.json())
    except requests.RequestException as exc:
        raise SystemExit(
            "FraudEye API is unavailable. Start Uvicorn first."
        ) from exc

    print("\nLoading transaction dataset...")
    df = pd.read_csv(DATA_PATH)

    
    # Select transactions for simulation
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "demo":
        fraud_rows = df[df["Class"] == 1]
        legit_rows = df[df["Class"] == 0]

        def make_batch():
            half = max(1, args.limit // 2)

            selected_fraud = fraud_rows.sample(
                n=half,
                replace=len(fraud_rows) < half,
            )
            selected_legit = legit_rows.sample(
                n=args.limit - half,
                replace=len(legit_rows) < args.limit - half,
            )

            return pd.concat(
                [selected_fraud, selected_legit]
            ).sample(frac=1).reset_index(drop=True)

    else:
        def make_batch():
            return df.sample(
                n=min(args.limit, len(df))
            ).reset_index(drop=True)

    stream = make_batch()


    fields = [
        "stream_id",
        "processed_at_utc",
        "dataset_row_index",
        "amount",
        "actual_label",
        "fraud_probability",
        "prediction",
        "alert",
        "threshold",
        "latency_ms",
        "api_status",
        "mode",
    ]

   
    counts = {
        "processed": 0,
        "alerts": 0,
        "api_errors": 0,
        "latencies": [],
    }

    # Continue stream IDs from the existing CSV
    stream_id = 0
    if OUTPUT_PATH.exists() and OUTPUT_PATH.stat().st_size > 0:
        try:
            old_ids = pd.read_csv(
                OUTPUT_PATH, usecols=["stream_id"]
            )["stream_id"]
            if not old_ids.empty:
                stream_id = int(
                    pd.to_numeric(
                        old_ids, errors="coerce"
                    ).max()
                )
        except Exception:
            stream_id = 0

    print("\n========== FraudEye AI Continuous Simulator ==========")
    print(f"Mode: {args.mode}")
    print("Continuous:", args.continuous)
    print("Press Ctrl+C to stop.\n")

    try:
        # Append mode preserves existing transaction records
        with OUTPUT_PATH.open(
            "a", newline="", encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file, fieldnames=fields
            )

            if file.tell() == 0:
                writer.writeheader()
                file.flush()

            while True:
                stream = make_batch()

                for _, row in stream.iterrows():
                    stream_id += 1

                    payload = {
                        f"V{i}": float(row[f"V{i}"])
                        for i in range(1, 29)
                    }
                    payload["Amount"] = float(row["Amount"])

                    request_start = perf_counter()

                    try:
                        response = requests.post(
                            API_URL,
                            json=payload,
                            timeout=10,
                        )
                        response.raise_for_status()
                        result = response.json()

                        latency_ms = (
                            perf_counter() - request_start
                        ) * 1000

                        probability = float(
                            result["fraud_probability"]
                        )
                        prediction = result["prediction"]
                        alert = result["alert"]
                        threshold = float(result["threshold"])

                        counts["processed"] += 1
                        counts["latencies"].append(latency_ms)

                        if str(prediction).lower() == "fraud":
                            counts["alerts"] += 1
                            print("      FRAUD ALERT!")

                        actual = (
                            "Fraud"
                            if int(row["Class"]) == 1
                            else "Legitimate"
                        )

                        record = {
                            "stream_id": stream_id,
                            "processed_at_utc": datetime.now(
                                timezone.utc
                            ).isoformat(),
                            "dataset_row_index": int(row.name),
                            "amount": float(row["Amount"]),
                            "actual_label": actual,
                            "fraud_probability": round(
                                probability, 6
                            ),
                            "prediction": prediction,
                            "alert": alert,
                            "threshold": threshold,
                            "latency_ms": round(latency_ms, 3),
                            "api_status": "success",
                            "mode": args.mode,
                        }

                        writer.writerow(record)
                        file.flush()

                        print(
                            f"[{stream_id:05d}] "
                            f"Amount={row['Amount']:.2f} | "
                            f"Score={probability:.2%} | "
                            f"Prediction={prediction} | "
                            f"Actual={actual} | "
                            f"Latency={latency_ms:.2f} ms"
                        )

                    except (
                        requests.RequestException,
                        KeyError,
                        TypeError,
                        ValueError,
                    ) as exc:
                        counts["api_errors"] += 1
                        latency_ms = (
                            perf_counter() - request_start
                        ) * 1000

                        writer.writerow({
                            "stream_id": stream_id,
                            "processed_at_utc": datetime.now(
                                timezone.utc
                            ).isoformat(),
                            "dataset_row_index": int(row.name),
                            "amount": float(row["Amount"]),
                            "actual_label": (
                                "Fraud"
                                if int(row["Class"]) == 1
                                else "Legitimate"
                            ),
                            "fraud_probability": "",
                            "prediction": "ERROR",
                            "alert": "API_ERROR",
                            "threshold": "",
                            "latency_ms": round(latency_ms, 3),
                            "api_status": "failed",
                            "mode": args.mode,
                        })
                        file.flush()

                        print(f"[{stream_id:05d}] API ERROR: {exc}")

                    time.sleep(args.delay)

                if not args.continuous:
                    break

    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")


    # Final summary
    print("\n========== Simulation Summary ==========")
    print(f"Successfully processed: {counts['processed']}")
    print(f"Fraud alerts: {counts['alerts']}")
    print(f"API errors: {counts['api_errors']}")

    if counts["latencies"]:
        latencies = counts["latencies"]
        print(f"Average API latency: {sum(latencies) / len(latencies):.2f} ms")
        print(f"Minimum API latency: {min(latencies):.2f} ms")
        print(f"Maximum API latency: {max(latencies):.2f} ms")

    print(f"CSV output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
