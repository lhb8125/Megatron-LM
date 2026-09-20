"""Shared command-line plumbing for JSON presets and explicit MoR overrides."""

from __future__ import annotations

import argparse
from pathlib import Path

from mor_mlite.config_loader import MoRPresetConfig, load_preset_config


def add_mor_config_arguments(parser: argparse.ArgumentParser) -> None:
    """Add model-shape overrides without exposing arbitrary parallel degrees."""

    parser.add_argument(
        "--preset-config",
        type=Path,
        help="compatible preset JSON; defaults to the checked-in configs/<preset>.json",
    )
    parser.add_argument("--n-start-layers", type=int)
    parser.add_argument("--n-recurrent-layers", type=int)
    parser.add_argument("--num-recursions", type=int)
    parser.add_argument("--n-end-layers", type=int)
    parser.add_argument(
        "--capacity-schedule",
        help="linear or an exact comma-separated capacity fraction for every recursion",
    )
    parser.add_argument("--router-temperature", type=float)
    parser.add_argument("--router-alpha", type=float)
    parser.add_argument("--router-aux-loss-coef", type=float)


def resolve_cli_preset(args: argparse.Namespace) -> MoRPresetConfig:
    return load_preset_config(args.preset, args.preset_config).with_overrides(
        n_start_layers=args.n_start_layers,
        n_recurrent_layers=args.n_recurrent_layers,
        num_recursions=args.num_recursions,
        n_end_layers=args.n_end_layers,
        capacity_schedule=args.capacity_schedule,
        router_temperature=args.router_temperature,
        router_alpha=args.router_alpha,
        router_aux_loss_coef=args.router_aux_loss_coef,
    )


__all__ = ["add_mor_config_arguments", "resolve_cli_preset"]
