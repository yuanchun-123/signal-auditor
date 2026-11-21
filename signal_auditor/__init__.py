"""signal-auditor: catch the usual ways a trading-signal backtest fools you."""

from .audit import AuditConfig, AuditReport, audit
from .checks import FAIL, PASS, WARN, CheckResult
from .panel import forward_returns, to_long, validate_panel

__all__ = ["audit", "AuditConfig", "AuditReport", "CheckResult", "PASS", "WARN", "FAIL",
           "forward_returns", "to_long", "validate_panel"]
__version__ = "0.1.0"
