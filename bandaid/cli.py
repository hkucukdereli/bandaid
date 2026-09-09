"""
Command-line entry points: ``bandaid-clean`` and ``bandaid-metadata``.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from bandaid._util import package_version
from bandaid.io import (
    load_tiff_stack,
    parse_filename,
    read_scanimage_metadata,
    save_session_metadata,
    save_tiff,
    write_json,
)
from bandaid.io.utils import get_files
from bandaid.ripple import detect_ripple, remove_ripple
from bandaid.scan_geometry import ScanGeometry

log = logging.getLogger("bandaid.cli")


def _setup_logging(verbose: int, quiet: bool) -> None:
    level = logging.WARNING if quiet else (logging.DEBUG if verbose > 1 else logging.INFO)
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr)


def _resolve_inputs(inputs: list[str]) -> list[Path]:
    files: list[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            files.extend(Path(f) for f, _ in get_files(p))
        elif p.is_file():
            files.append(p)
        else:
            raise FileNotFoundError(item)
    seen, unique = set(), []
    for f in files:
        if f.resolve() not in seen:
            seen.add(f.resolve())
            unique.append(f)
    return unique


def _session_header(files: list[Path]) -> dict:
    folder = files[0].parent
    parsed = parse_filename(folder.name)
    return {
        "session_dir": str(folder),
        "mouse": parsed["mouse"],
        "date": parsed["date"],
        "n_files": len(files),
        "bandaid_version": package_version(),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def _print_detection_table(rows: list[dict]) -> None:
    hdr = f"{'file':40s} {'ripple':7s} {'n':>4s} {'f0 kHz':>7s} {'SNR dB':>7s} {'harm':>4s} {'amp':>6s} {'cyc/frame':>9s}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['file'][:40]:40s} {'yes' if r['detected'] else 'no':7s} {str(r['n']):>4s} "
            f"{(r['f0_hz'] or 0) / 1e3:7.1f} {r['snr_db']:7.1f} {r['harmonics']:4d} "
            f"{r['amplitude']:6.1f} {r['advance']:9.4f}"
        )


def _clean_pca(path: Path, out: Path, args, geometry: ScanGeometry | None) -> dict:
    from bandaid.pca_clean import remove_ripple_pca

    t0 = time.perf_counter()
    stack = load_tiff_stack(path)
    cleaned, info = remove_ripple_pca(
        stack,
        n_noise_pcs=args.n_noise_pcs,
        auto_detect=not args.no_auto_detect,
        fit_frames=args.pca_fit_frames,
        return_info=True,
    )
    save_tiff(cleaned, out, bigtiff=True)
    return {
        "source": str(path),
        "output": str(out),
        "method": "pca",
        "n_frames": int(stack.shape[0]),
        "shape": list(stack.shape[1:]),
        "dtype": str(stack.dtype),
        "noise_pcs": [int(i) for i in info["noise_pcs"]],
        "explained_variance_ratio": [float(v) for v in info["explained_variance_ratio"]],
        "removed": bool(info["noise_pcs"]),
        "runtime_s": {"total": time.perf_counter() - t0},
    }


def _write_diagnostics_for(path: Path, out: Path, report, out_dir: Path) -> None:
    from bandaid.diagnostics import write_diagnostics

    fit = report.fits[0] if report.fits else None
    if fit is None:
        return
    n = report.n_frames
    a = max(0, n // 2 - 150)
    b = min(n, a + 300)
    raw = load_tiff_stack(path, frames=range(a, b), channel=0 if report.n_channels > 1 else None)
    cleaned = load_tiff_stack(out, frames=range(a, b), channel=0 if report.n_channels > 1 else None)
    if raw.ndim == 4:
        raw, cleaned = raw[:, 0], cleaned[:, 0]
    geometry = ScanGeometry.from_si_params(read_scanimage_metadata(path, frames=False)["si_params"])
    paths = write_diagnostics(raw, cleaned, fit, geometry, out_dir / "diagnostics", path.stem, frame_offset=a)
    log.info("diagnostics: %s", ", ".join(p.name for p in paths))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bandaid-clean",
        description="Detect and remove moving periodic banding (PMT ripple) from ScanImage TIFF movies.",
    )
    parser.add_argument("inputs", nargs="+", help="session folder(s) and/or TIFF file(s)")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="output folder (default: <session>/cleaned)")
    parser.add_argument("--method", choices=["ripple", "pca"], default="ripple",
                        help="ripple: scan-time harmonic model (default); pca: PCA subtraction (in memory)")
    parser.add_argument("--harmonics", type=int, default=8, help="harmonics in the ripple model (default 8)")
    parser.add_argument("--row-groups", type=int, default=8, help="row groups for the profile fit (default 8)")
    parser.add_argument("--window-frames", type=int, default=256, help="least-squares window (default 256)")
    parser.add_argument("--chunk-frames", type=int, default=None, help="frames per streaming chunk")
    parser.add_argument("--no-mask", action="store_true", help="do not exclude bright pixels from the fit")
    parser.add_argument("--dry-run", action="store_true", help="only detect; print a table, write nothing")
    parser.add_argument("--force", action="store_true", help="remove even if detection is negative")
    parser.add_argument("--copy-clean", action="store_true",
                        help="copy files without detected ripple to the output folder unchanged")
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing outputs")
    parser.add_argument("--diagnostics", action="store_true", help="save diagnostic PNGs next to the outputs")
    parser.add_argument("--report", default="ripple_report.json", help="report filename in the output folder")
    parser.add_argument("--n-noise-pcs", type=int, default=None, help="[pca] number of leading PCs to remove")
    parser.add_argument("--no-auto-detect", action="store_true", help="[pca] disable PC auto-detection")
    parser.add_argument("--pca-fit-frames", type=int, default=2000, help="[pca] frames used to fit the PCA")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    parser.add_argument("-q", "--quiet", action="store_true")
    parser.add_argument("--version", action="version", version=f"bandaid {package_version()}")
    args = parser.parse_args(argv)
    _setup_logging(args.verbose, args.quiet)

    files = _resolve_inputs(args.inputs)
    if not files:
        parser.error("no TIFF files found")
    out_dir = args.out_dir or (files[0].parent / "cleaned")

    if args.dry_run:
        rows = []
        for path in files:
            det = detect_ripple(path)
            rows.append({
                "file": path.name, "detected": det.detected, "n": det.harmonic_index,
                "f0_hz": det.f0_hz, "snr_db": det.snr_db, "harmonics": det.n_harmonics_detected,
                "amplitude": det.amplitude_counts, "advance": det.phase_advance_cycles_per_frame,
            })
        _print_detection_table(rows)
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    reports, failures = [], 0
    for path in files:
        out = out_dir / path.name
        if out.exists() and not args.overwrite:
            log.warning("%s exists; skipping (use --overwrite)", out)
            continue
        try:
            if args.method == "pca":
                reports.append(_clean_pca(path, out, args, None))
                continue
            _, report = remove_ripple(
                path, out=out, n_harmonics=args.harmonics, n_row_groups=args.row_groups,
                fit_mask=None if args.no_mask else "auto", window_frames=args.window_frames,
                chunk_frames=args.chunk_frames, force=args.force, copy_if_clean=args.copy_clean,
            )
            d = report.to_dict()
            d["method"] = "ripple"
            reports.append(d)
            if args.diagnostics and report.removed:
                _write_diagnostics_for(path, out, report, out_dir)
            log.info("%s -> %s (%s, %.1f s)", path.name, out if report.removed else "not written",
                     "cleaned" if report.removed else "no ripple", report.runtime_s["total"])
        except Exception as exc:  # keep going with the other files
            failures += 1
            log.error("%s failed: %s", path.name, exc, exc_info=args.verbose > 0)
            reports.append({"source": str(path), "error": str(exc)})
    report_path = out_dir / args.report
    write_json({"session": _session_header(files), "files": reports}, report_path)
    log.info("report written to %s", report_path)
    return 1 if failures else 0


def metadata_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bandaid-metadata",
        description="Collect ScanImage metadata of a session folder into session_metadata.json.",
    )
    parser.add_argument("session_dirs", nargs="+", help="session folder(s) containing TIFF files")
    parser.add_argument("--output", type=Path, default=None,
                        help="output JSON path (single session only; default <session>/session_metadata.json)")
    parser.add_argument("--no-frames", action="store_true", help="skip per-frame metadata (faster)")
    parser.add_argument("--mouse", default=None)
    parser.add_argument("--date", default=None)
    parser.add_argument("--session-id", default=None)
    parser.add_argument("-v", "--verbose", action="count", default=0)
    args = parser.parse_args(argv)
    _setup_logging(args.verbose, False)
    if args.output and len(args.session_dirs) > 1:
        parser.error("--output can only be used with a single session folder")
    for d in args.session_dirs:
        path = save_session_metadata(d, output_path=args.output, mouse=args.mouse, date=args.date,
                                     session_id=args.session_id, frames=not args.no_frames)
        print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
