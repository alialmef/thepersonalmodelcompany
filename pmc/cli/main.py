"""Top-level wiring for `pmc chat`, `pmc configure`, `pmc whoami`.

These three subcommands are mounted into the existing `pmc` CLI parser
in pmc/orchestrator/cli.py. The split keeps the new "talk to your
agent" surface separate from the old training-pipeline code so future
refactors of either don't drag the other along.
"""

from __future__ import annotations

import argparse
import sys

from pmc.cli.configure import cmd_configure, cmd_show
from pmc.cli.local_config import CONFIG_FILE, load


def cmd_chat(args: argparse.Namespace) -> int:
    cfg = load()
    if cfg is None:
        print(f"no config found at {CONFIG_FILE}.")
        print("run `pmc configure` to set up your provider, then `pmc chat` again.")
        return 1
    from pmc.cli.chat import run
    return run(
        cfg,
        skip_opener=bool(getattr(args, "no_opener", False)),
        clear=not bool(getattr(args, "no_clear", False)),
    )


def cmd_whoami(args: argparse.Namespace) -> int:
    cfg = load()
    if cfg is None:
        print("(unconfigured)")
        return 1
    from pmc.cli.context import build_context
    storage = cfg.effective_storage_root()
    user_id = cfg.user_id or "local"
    ctx = build_context(storage, user_id)
    sys.stdout.write(ctx)
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    """Attach the new subcommands to the existing pmc parser."""
    chat = sub.add_parser("chat", help="Talk to your agent in the terminal")
    chat.add_argument("--no-opener", action="store_true",
                      help="Skip the auto-fired opener turn")
    chat.add_argument("--no-clear", action="store_true",
                      help="Don't clear the screen on chat start "
                      "(preserve pre-chat terminal scrollback)")
    chat.set_defaults(func=cmd_chat)

    cfg = sub.add_parser("configure", help="Set provider, model, and API key")
    cfg.add_argument("--provider", help="anthropic | openai | google | openrouter")
    cfg.add_argument("--model")
    cfg.add_argument("--api-key")
    cfg.add_argument("--user-id")
    cfg.add_argument("--user-email")
    cfg.add_argument("--storage-root")
    cfg.add_argument("--no-validate", action="store_true",
                     help="Skip the key-validity probe")
    cfg.add_argument("--no-ingest", action="store_true",
                     help="Don't offer to populate the graph at the end")
    cfg.set_defaults(func=cmd_configure)

    show = sub.add_parser("config-show", help="Print the saved CLI config")
    show.set_defaults(func=cmd_show)

    me = sub.add_parser("whoami", help="Print the context block the agent sees")
    me.set_defaults(func=cmd_whoami)

    # Engine parity: `pmc ingest` populates the graph by calling the
    # same Rust extractors the Mac app does. CLI users get a complete
    # graph without needing to install/run the GUI.
    from pmc.cli.ingest import register as register_ingest
    register_ingest(sub)

    # `pmc time` — inspect the time model (the bedrock of the portrait).
    from pmc.cli.time_cmd import register as register_time
    register_time(sub)

    # `pmc reading` — what you've been reading (textural layer).
    from pmc.cli.reading_cmd import register as register_reading
    register_reading(sub)

    # `pmc serve --mcp` + `pmc install-mcp <agent>` — expose PMC's
    # substrate to any MCP-speaking agent.
    from pmc.cli.serve_cmd import register as register_serve
    register_serve(sub)

    # `pmc sandbox` — fresh-start iteration harness.
    from pmc.cli.sandbox import register as register_sandbox
    register_sandbox(sub)

    # `pmc doctor` — preflight diagnostic for fresh users.
    from pmc.cli.doctor import register as register_doctor
    register_doctor(sub)

    # Teaching the gate: pmc remember / pmc forget write rows into
    # ~/.pmc/gate-feedback.jsonl that override the default rules.
    from pmc.cli.learn import register as register_learn
    register_learn(sub)

    # The Gate: `pmc watch` is the long-running daemon that catches
    # filesystem / DB / distributed-notification events and routes them
    # through the classifier. Lives in pmc/watch/.
    watch = sub.add_parser(
        "watch",
        help="Run the long-lived daemon — watch the Mac for changes "
        "and react in real time",
    )
    watch.add_argument("--log-file", action="store_true",
                       help="write logs to ~/.pmc/watch.log instead of stderr "
                       "(used by the launchd plist)")
    watch.add_argument("-v", "--verbose", action="store_true",
                       help="DEBUG-level logging")
    watch.set_defaults(func=cmd_watch)

    # Launchd install / uninstall / status — pmc watch-install etc.
    _register_watch_admin(sub)

    # `pmc consolidate` — fire one consolidator pass manually
    consol = sub.add_parser(
        "consolidate",
        help="Run one memory-consolidation pass over the graph",
    )
    consol.add_argument(
        "--only", action="append", metavar="LAYER",
        help="re-run only specific layers (time, work, attention, "
             "interests, reading, time_md, self_md, whoami). "
             "Can be passed multiple times.",
    )
    consol.add_argument(
        "--rebuild", action="store_true",
        help="wipe portrait/ before running so the pass starts fresh "
             "(no prior-self.md feedback)",
    )
    consol.add_argument(
        "--diff", action="store_true",
        help="snapshot the current portrait before running, then print "
             "what changed afterward",
    )
    consol.set_defaults(func=cmd_consolidate)

    # `pmc rebuild` — shortcut for --rebuild --diff
    rb = sub.add_parser(
        "rebuild",
        help="Wipe portrait/ and rebuild from scratch with a diff "
             "(shortcut for `pmc consolidate --rebuild --diff`)",
    )
    rb.add_argument("--only", action="append", metavar="LAYER")
    rb.set_defaults(func=lambda a: cmd_consolidate(argparse.Namespace(
        only=a.only, rebuild=True, diff=True,
    )))


def cmd_consolidate(args: argparse.Namespace) -> int:
    cfg = load()
    if cfg is None:
        print("no config — run pmc configure first.")
        return 1
    from rich.console import Console
    from pmc.cli import ui
    from pmc.consolidator import run_consolidation
    from pmc.consolidator.run import diff_against_latest_snapshot

    console = Console()
    console.print()
    only = getattr(args, "only", None) or None
    rebuild = bool(getattr(args, "rebuild", False))
    diff = bool(getattr(args, "diff", False))

    if rebuild:
        ui.say_dim(console, "rebuilding from scratch (portrait/ will be wiped)…")
    elif only:
        ui.say_dim(console, f"running consolidator (only: {', '.join(only)})…")
    else:
        ui.say_dim(console, "running consolidator…")

    result = run_consolidation(cfg, only=only, rebuild=rebuild, diff=diff)
    console.print()
    ui.say(console, f"{ui.GLYPH_DONE} consolidated  ·  {result.duration_s:.2f}s",
           style=ui.OK)
    ui.say_dim(console, result.notes)

    if diff:
        portrait_dir = (
            cfg.effective_storage_root() / "users" / (cfg.user_id or "local")
            / "graph" / "synth" / "portrait"
        )
        console.print()
        ui.card_title(console, "diff vs prior snapshot")
        diff_text = diff_against_latest_snapshot(portrait_dir)
        for line in diff_text.splitlines()[:80]:
            console.print(f"{ui.margin()}{line}")
    return 0


def cmd_watch(args: argparse.Namespace) -> int:
    from pmc.watch.daemon import run as watch_run
    return watch_run(log_to_file=args.log_file, verbose=args.verbose)


def _register_watch_admin(sub: argparse._SubParsersAction) -> None:
    from pmc.watch.launchd import register as register_launchd
    register_launchd(sub)


__all__ = ["register"]
