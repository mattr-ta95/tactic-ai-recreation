#!/usr/bin/env python3
"""
Download and validate ALL StatsBomb open data for TacticAI.

Two-phase pipeline:
  --validate   Sample 2 matches per competition, report freeze frame availability
  --download   Download all validated competitions, merge with existing data

Usage:
  python scripts/download_all_statsbomb.py --validate
  python scripts/download_all_statsbomb.py --download
  python scripts/download_all_statsbomb.py --validate --download
  python scripts/download_all_statsbomb.py --download --include-women
"""

import argparse
import ast
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
from statsbombpy import sb

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALREADY_DOWNLOADED = {
    (43, 3),    # FIFA World Cup 2018
    (43, 106),  # FIFA World Cup 2022
    (55, 43),   # UEFA Euro 2020
    (55, 282),  # UEFA Euro 2024
}

WOMEN_COMPETITION_IDS = {37, 49, 53, 72}  # FA WSL, NWSL, UEFA Women's Euro, Women's WC

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
VALIDATION_FILE = DATA_DIR / "validation_results.json"

SHOTS_FILE = PROCESSED_DIR / "shots_freeze.pkl"
CORNERS_FILE = PROCESSED_DIR / "corners.pkl"
EVENTS_FILE = RAW_DIR / "events_multi_league.csv"
MATCHES_FILE = RAW_DIR / "matches_multi_league.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_freeze_frame(ff: Any) -> Optional[List[Dict]]:
    """Parse a freeze frame value into a list of player dicts, or None."""
    if ff is None:
        return None
    if isinstance(ff, list):
        return ff if len(ff) > 0 else None
    if not isinstance(ff, (list, dict)):
        try:
            if pd.isna(ff):
                return None
        except (TypeError, ValueError):
            pass
    if isinstance(ff, str):
        try:
            result = ast.literal_eval(ff)
            return result if result else None
        except (ValueError, SyntaxError):
            return None
    return ff if ff else None


def _competition_key(row: pd.Series) -> Tuple[int, int]:
    return (int(row["competition_id"]), int(row["season_id"]))


def _is_women(row: pd.Series) -> bool:
    return (
        int(row["competition_id"]) in WOMEN_COMPETITION_IDS
        or row.get("competition_gender") == "female"
    )


def _load_competitions(include_women: bool) -> pd.DataFrame:
    """Load StatsBomb competitions, filtering already-downloaded and optionally women's."""
    comps = sb.competitions()
    # Drop already downloaded
    comps = comps[
        ~comps.apply(lambda r: _competition_key(r) in ALREADY_DOWNLOADED, axis=1)
    ]
    if not include_women:
        comps = comps[~comps.apply(_is_women, axis=1)]
    return comps.reset_index(drop=True)


def _safe_fetch_events(match_id: int) -> Optional[pd.DataFrame]:
    """Fetch events for a single match, returning None on failure."""
    try:
        return sb.events(match_id=match_id)
    except Exception as exc:
        log.warning("Failed to fetch events for match %s: %s", match_id, exc)
        return None


# ---------------------------------------------------------------------------
# Phase 1: Validate
# ---------------------------------------------------------------------------


def validate(
    include_women: bool = False,
    sample_size: int = 2,
    min_ff_rate: float = 0.5,
) -> List[Dict]:
    """Sample matches per competition and report freeze frame availability."""
    log.info("Phase 1: VALIDATE — sampling %d matches per competition", sample_size)
    comps = _load_competitions(include_women)
    log.info("Evaluating %d competition-seasons (skipping %d already downloaded)", len(comps), len(ALREADY_DOWNLOADED))

    results: List[Dict] = []

    for idx, (_, comp) in enumerate(comps.iterrows()):
        comp_id = int(comp["competition_id"])
        season_id = int(comp["season_id"])
        comp_name = comp["competition_name"]
        season_name = comp["season_name"]
        gender = comp.get("competition_gender", "male")
        has_360_col = comp.get("match_available_360")

        log.info(
            "[%d/%d] %s %s (comp=%d, season=%d)",
            idx + 1, len(comps), comp_name, season_name, comp_id, season_id,
        )

        try:
            matches = sb.matches(competition_id=comp_id, season_id=season_id)
        except Exception as exc:
            log.warning("  Could not fetch matches: %s", exc)
            results.append({
                "competition_id": comp_id,
                "season_id": season_id,
                "competition_name": comp_name,
                "season_name": season_name,
                "gender": gender,
                "total_matches": 0,
                "sampled": 0,
                "total_shots": 0,
                "shots_with_ff": 0,
                "corner_shots": 0,
                "ff_rate": 0.0,
                "estimated_corners": 0,
                "has_360": bool(has_360_col),
                "passed": False,
                "error": str(exc),
            })
            continue

        total_matches = len(matches)
        if total_matches == 0:
            log.info("  No matches found, skipping")
            results.append({
                "competition_id": comp_id,
                "season_id": season_id,
                "competition_name": comp_name,
                "season_name": season_name,
                "gender": gender,
                "total_matches": 0,
                "sampled": 0,
                "total_shots": 0,
                "shots_with_ff": 0,
                "corner_shots": 0,
                "ff_rate": 0.0,
                "estimated_corners": 0,
                "has_360": bool(has_360_col),
                "passed": False,
            })
            continue

        # Sample random matches
        sample_ids = matches["match_id"].tolist()
        random.seed(42)  # Reproducible sampling
        sample_ids = random.sample(sample_ids, min(sample_size, len(sample_ids)))

        total_shots = 0
        shots_with_ff = 0
        corner_shots = 0
        sampled = 0

        for mid in sample_ids:
            events = _safe_fetch_events(mid)
            if events is None:
                continue
            sampled += 1
            shots = events[events["type"] == "Shot"]
            total_shots += len(shots)
            if "shot_freeze_frame" in events.columns:
                ff_shots = shots[shots["shot_freeze_frame"].notna()]
                shots_with_ff += len(ff_shots)
                # Corner shots: play_pattern == 'From Corner'
                if "play_pattern" in events.columns:
                    corner_shots += len(
                        shots[shots["play_pattern"] == "From Corner"]
                    )

        ff_rate = shots_with_ff / total_shots if total_shots > 0 else 0.0
        # Estimate corners for entire competition using sampled rate
        avg_corners_per_match = corner_shots / sampled if sampled > 0 else 0.0
        estimated_corners = int(avg_corners_per_match * total_matches)
        passed = ff_rate >= min_ff_rate and sampled > 0

        log.info(
            "  %d matches, sampled %d: %d/%d shots have FF (%.0f%%), ~%d est. corners → %s",
            total_matches, sampled, shots_with_ff, total_shots,
            ff_rate * 100, estimated_corners, "PASS" if passed else "SKIP",
        )

        results.append({
            "competition_id": comp_id,
            "season_id": season_id,
            "competition_name": comp_name,
            "season_name": season_name,
            "gender": gender,
            "total_matches": total_matches,
            "sampled": sampled,
            "total_shots": total_shots,
            "shots_with_ff": shots_with_ff,
            "corner_shots": corner_shots,
            "ff_rate": round(ff_rate, 3),
            "estimated_corners": estimated_corners,
            "has_360": bool(has_360_col),
            "passed": passed,
        })

    # Save results
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    VALIDATION_FILE.write_text(json.dumps(results, indent=2))
    log.info("Saved validation results to %s", VALIDATION_FILE)

    # Print summary table
    _print_validation_table(results)
    return results


def _print_validation_table(results: List[Dict]) -> None:
    """Print a formatted summary table of validation results."""
    header = f"{'Competition':<35} {'Season':<12} {'Matches':>7} {'FF Rate':>8} {'Est.Corners':>12} {'360?':>5} {'Status':>7}"
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)

    total_matches = 0
    total_corners = 0
    passed_count = 0

    for r in sorted(results, key=lambda x: x["estimated_corners"], reverse=True):
        status = "PASS" if r["passed"] else "SKIP"
        ff_pct = f"{r['ff_rate']*100:.0f}%" if r["total_shots"] > 0 else "N/A"
        has_360 = "Yes" if r["has_360"] else "No"
        print(
            f"{r['competition_name']:<35} {r['season_name']:<12} {r['total_matches']:>7} "
            f"{ff_pct:>8} {r['estimated_corners']:>12} {has_360:>5} {status:>7}"
        )
        if r["passed"]:
            total_matches += r["total_matches"]
            total_corners += r["estimated_corners"]
            passed_count += 1

    print(sep)
    print(
        f"{'TOTAL DOWNLOADABLE':<35} {'':12} {total_matches:>7} "
        f"{'':>8} {total_corners:>12} {'':>5} {passed_count:>4} comps"
    )
    print(sep)


# ---------------------------------------------------------------------------
# Phase 2: Download
# ---------------------------------------------------------------------------


def download(
    include_women: bool = False,
    min_ff_rate: float = 0.5,
) -> None:
    """Download all validated competitions and merge with existing data."""
    # Load validation results
    if not VALIDATION_FILE.exists():
        log.error(
            "Validation results not found at %s. Run --validate first.", VALIDATION_FILE
        )
        return

    results = json.loads(VALIDATION_FILE.read_text())
    passed = [r for r in results if r["passed"]]
    if not include_women:
        passed = [
            r for r in passed
            if r["gender"] != "female"
            and int(r["competition_id"]) not in WOMEN_COMPETITION_IDS
        ]

    if not passed:
        log.error("No competitions passed validation. Run --validate first.")
        return

    log.info(
        "Phase 2: DOWNLOAD — %d competitions passed validation", len(passed)
    )
    for r in passed:
        log.info(
            "  %s %s (%d matches, ~%d corners)",
            r["competition_name"], r["season_name"],
            r["total_matches"], r["estimated_corners"],
        )

    # Ensure directories exist
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing data for incremental merge
    existing_event_ids: Set[str] = set()
    if SHOTS_FILE.exists():
        existing_shots = pd.read_pickle(SHOTS_FILE)
        existing_event_ids.update(existing_shots["id"].astype(str).tolist())
        log.info("Loaded %d existing shots (will deduplicate)", len(existing_shots))
    else:
        existing_shots = pd.DataFrame()

    existing_corner_ids: Set[str] = set()
    if CORNERS_FILE.exists():
        existing_corners = pd.read_pickle(CORNERS_FILE)
        existing_corner_ids.update(existing_corners["id"].astype(str).tolist())
        log.info("Loaded %d existing corners", len(existing_corners))
    else:
        existing_corners = pd.DataFrame()

    # Track what we download
    all_new_events: List[pd.DataFrame] = []
    all_new_matches: List[pd.DataFrame] = []
    comp_stats: List[Dict] = []
    total_match_count = 0
    total_event_count = 0

    for comp_info in passed:
        comp_id = int(comp_info["competition_id"])
        season_id = int(comp_info["season_id"])
        comp_name = comp_info["competition_name"]
        season_name = comp_info["season_name"]

        log.info("=" * 70)
        log.info("Downloading: %s %s (comp=%d, season=%d)", comp_name, season_name, comp_id, season_id)

        try:
            matches = sb.matches(competition_id=comp_id, season_id=season_id)
        except Exception as exc:
            log.error("  Failed to fetch matches: %s", exc)
            continue

        log.info("  %d matches to download", len(matches))
        matches["competition_id"] = comp_id
        matches["season_id"] = season_id
        matches["competition_name"] = comp_name
        matches["season_desc"] = season_name
        all_new_matches.append(matches)

        comp_events: List[pd.DataFrame] = []
        successful = 0
        failed = 0

        for i, (_, match) in enumerate(matches.iterrows()):
            match_id = match["match_id"]
            events = _safe_fetch_events(match_id)
            if events is None:
                failed += 1
                continue

            events["match_id"] = match_id
            events["competition_id"] = comp_id
            events["season_id"] = season_id
            events["competition_name"] = comp_name
            events["season_desc"] = season_name
            comp_events.append(events)
            successful += 1

            if (i + 1) % 10 == 0:
                log.info(
                    "  [%d/%d] %d OK, %d failed",
                    i + 1, len(matches), successful, failed,
                )

        if not comp_events:
            log.warning("  No events downloaded for %s %s", comp_name, season_name)
            continue

        events_df = pd.concat(comp_events, ignore_index=True)
        all_new_events.append(events_df)

        # Per-competition stats
        shots = events_df[events_df["type"] == "Shot"]
        shots_ff = (
            shots[shots["shot_freeze_frame"].notna()]
            if "shot_freeze_frame" in events_df.columns
            else pd.DataFrame()
        )
        corners = events_df[
            (events_df["type"] == "Pass")
            & (events_df.get("pass_type", pd.Series(dtype=str)) == "Corner")
        ] if "pass_type" in events_df.columns else pd.DataFrame()

        log.info(
            "  Done: %d/%d matches, %d events, %d shots (%d with FF), %d corner passes",
            successful, len(matches), len(events_df), len(shots),
            len(shots_ff), len(corners),
        )
        comp_stats.append({
            "competition": f"{comp_name} {season_name}",
            "matches": successful,
            "events": len(events_df),
            "shots": len(shots),
            "shots_with_ff": len(shots_ff),
            "corners": len(corners),
        })
        total_match_count += successful
        total_event_count += len(events_df)

    if not all_new_events:
        log.error("No new events downloaded.")
        return

    # Combine all new events
    log.info("=" * 70)
    log.info("POST-PROCESSING: Merging new data with existing")
    new_events = pd.concat(all_new_events, ignore_index=True)
    log.info("Total new events: %d from %d matches", len(new_events), total_match_count)

    # --- Extract and merge shots ---
    new_shots = new_events[
        (new_events["type"] == "Shot")
        & (new_events["shot_freeze_frame"].notna())
    ].copy()
    log.info("New shots with freeze frames: %d", len(new_shots))

    # Parse freeze frames
    new_shots["freeze_frame_parsed"] = new_shots["shot_freeze_frame"].apply(
        parse_freeze_frame
    )
    new_shots = new_shots[new_shots["freeze_frame_parsed"].notna()].copy()
    log.info("After parsing freeze frames: %d valid", len(new_shots))

    # Deduplicate against existing
    new_shots = new_shots[~new_shots["id"].astype(str).isin(existing_event_ids)].copy()
    log.info("After dedup: %d genuinely new shots", len(new_shots))

    combined_shots = pd.concat(
        [existing_shots, new_shots], ignore_index=True
    )
    # Final dedup safety net
    combined_shots = combined_shots.drop_duplicates(subset=["id"], keep="first")
    combined_shots.to_pickle(SHOTS_FILE)
    log.info("Saved %d total shots to %s", len(combined_shots), SHOTS_FILE)

    # --- Extract and merge corners ---
    new_corners = pd.DataFrame()
    if "pass_type" in new_events.columns:
        new_corners = new_events[
            (new_events["type"] == "Pass")
            & (new_events["pass_type"] == "Corner")
        ].copy()
    log.info("New corner passes: %d", len(new_corners))

    if len(new_corners) > 0:
        new_corners = new_corners[
            ~new_corners["id"].astype(str).isin(existing_corner_ids)
        ].copy()
        log.info("After dedup: %d genuinely new corners", len(new_corners))

    combined_corners = pd.concat(
        [existing_corners, new_corners], ignore_index=True
    )
    combined_corners = combined_corners.drop_duplicates(subset=["id"], keep="first")
    combined_corners.to_pickle(CORNERS_FILE)
    log.info("Saved %d total corners to %s", len(combined_corners), CORNERS_FILE)

    # --- Append events CSV (incremental, deduplicated) ---
    log.info("Appending events to %s ...", EVENTS_FILE)
    if EVENTS_FILE.exists():
        # Load only the id column for dedup (memory-efficient for large CSVs)
        try:
            existing_csv_ids = set(
                pd.read_csv(EVENTS_FILE, usecols=["id"], dtype=str)["id"].tolist()
            )
            new_events_deduped = new_events[
                ~new_events["id"].astype(str).isin(existing_csv_ids)
            ]
            log.info(
                "Filtered %d → %d new events (removed %d duplicates)",
                len(new_events), len(new_events_deduped),
                len(new_events) - len(new_events_deduped),
            )
            new_events_deduped.to_csv(EVENTS_FILE, mode="a", header=False, index=False)
            log.info("Appended %d new events to CSV", len(new_events_deduped))
        except Exception as exc:
            log.warning("Could not dedup events CSV (%s), appending all", exc)
            new_events.to_csv(EVENTS_FILE, mode="a", header=False, index=False)
            log.info("Appended %d events to CSV (no dedup)", len(new_events))
    else:
        new_events.to_csv(EVENTS_FILE, mode="w", header=True, index=False)
        log.info("Created events CSV with %d events", len(new_events))

    # --- Append matches CSV ---
    if all_new_matches:
        new_matches_df = pd.concat(all_new_matches, ignore_index=True)
        if MATCHES_FILE.exists():
            try:
                existing_match_ids = set(
                    pd.read_csv(MATCHES_FILE, usecols=["match_id"], dtype=str)["match_id"].tolist()
                )
                new_matches_df = new_matches_df[
                    ~new_matches_df["match_id"].astype(str).isin(existing_match_ids)
                ]
            except Exception:
                pass  # If dedup fails, append all
            new_matches_df.to_csv(MATCHES_FILE, mode="a", header=False, index=False)
        else:
            new_matches_df.to_csv(MATCHES_FILE, mode="w", header=True, index=False)
        log.info("Saved %d new matches to %s", len(new_matches_df), MATCHES_FILE)

    # --- Data integrity checks ---
    log.info("=" * 70)
    log.info("DATA INTEGRITY CHECKS")
    _run_integrity_checks(combined_shots, combined_corners)

    # --- Final summary ---
    log.info("=" * 70)
    log.info("DOWNLOAD SUMMARY")
    log.info("=" * 70)
    for s in comp_stats:
        log.info(
            "  %-40s %3d matches  %6d shots (%5d FF)  %4d corners",
            s["competition"], s["matches"], s["shots"], s["shots_with_ff"], s["corners"],
        )
    log.info("-" * 70)
    log.info("  Total new: %d matches, %d events", total_match_count, total_event_count)
    log.info("  Combined shots_freeze.pkl: %d rows", len(combined_shots))
    log.info("  Combined corners.pkl: %d rows", len(combined_corners))
    log.info("=" * 70)
    log.info("Next steps:")
    log.info("  python scripts/prepare_training_data.py")
    log.info("  python scripts/generate_synthetic_corners.py --combine")
    log.info("  python scripts/train_baseline.py")


def _run_integrity_checks(shots: pd.DataFrame, corners: pd.DataFrame) -> None:
    """Validate merged data integrity."""
    issues = 0

    # 1. No duplicate event IDs in shots
    dupes = shots["id"].duplicated().sum()
    if dupes > 0:
        log.warning("  FAIL: %d duplicate event IDs in shots_freeze.pkl", dupes)
        issues += 1
    else:
        log.info("  PASS: No duplicate event IDs in shots")

    # 2. No duplicate event IDs in corners
    dupes_c = corners["id"].duplicated().sum()
    if dupes_c > 0:
        log.warning("  FAIL: %d duplicate event IDs in corners.pkl", dupes_c)
        issues += 1
    else:
        log.info("  PASS: No duplicate event IDs in corners")

    # 3. All shots have valid freeze_frame_parsed
    if "freeze_frame_parsed" in shots.columns:
        invalid_ff = shots["freeze_frame_parsed"].isna().sum()
        if invalid_ff > 0:
            log.warning("  FAIL: %d shots missing freeze_frame_parsed", invalid_ff)
            issues += 1
        else:
            log.info("  PASS: All shots have freeze_frame_parsed")

        # 4. Check freeze frame structure (sample)
        sample = shots["freeze_frame_parsed"].dropna().head(100)
        bad_structure = 0
        for ff in sample:
            if not isinstance(ff, list):
                bad_structure += 1
                continue
            for player in ff:
                if not isinstance(player, dict):
                    bad_structure += 1
                    break
                if "location" not in player or "teammate" not in player:
                    bad_structure += 1
                    break
        if bad_structure > 0:
            log.warning(
                "  WARN: %d/%d sampled freeze frames have unexpected structure",
                bad_structure, len(sample),
            )
        else:
            log.info("  PASS: Freeze frame structure looks correct (sampled %d)", len(sample))

    # 5. Competition metadata populated
    if "competition_name" in shots.columns:
        missing_comp = shots["competition_name"].isna().sum()
        if missing_comp > 0:
            log.warning("  WARN: %d shots missing competition_name", missing_comp)
        else:
            log.info("  PASS: All shots have competition metadata")

        # Breakdown by competition
        log.info("  Shots by competition:")
        for comp, count in shots["competition_name"].value_counts().items():
            log.info("    %-30s %d", comp, count)

    if issues == 0:
        log.info("  All integrity checks passed!")
    else:
        log.warning("  %d integrity issues found — review above", issues)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download & validate ALL StatsBomb open data for TacticAI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/download_all_statsbomb.py --validate
  python scripts/download_all_statsbomb.py --download
  python scripts/download_all_statsbomb.py --validate --download
  python scripts/download_all_statsbomb.py --download --include-women
        """,
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Sample 2 matches per competition, report freeze frame availability",
    )
    parser.add_argument(
        "--download", action="store_true",
        help="Download all validated competitions (reads validation_results.json)",
    )
    parser.add_argument(
        "--include-women", action="store_true",
        help="Include women's competitions (FA WSL, NWSL, Women's WC, etc.)",
    )
    parser.add_argument(
        "--min-ff-rate", type=float, default=0.5,
        help="Minimum freeze frame rate to include a competition (default: 0.5)",
    )
    parser.add_argument(
        "--sample-size", type=int, default=2,
        help="Matches to sample per competition during validation (default: 2)",
    )

    args = parser.parse_args()

    if not args.validate and not args.download:
        parser.error("Specify --validate, --download, or both")

    if args.validate:
        validate(
            include_women=args.include_women,
            sample_size=args.sample_size,
            min_ff_rate=args.min_ff_rate,
        )

    if args.download:
        download(
            include_women=args.include_women,
            min_ff_rate=args.min_ff_rate,
        )


if __name__ == "__main__":
    main()
