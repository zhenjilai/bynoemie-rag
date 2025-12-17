"""
Logger Utility

Provides structured logging with context support.
"""

import os
import logging
import logging.config
from typing import Any, Dict, Optional
from pathlib import Path
from functools import wraps
import time
import json


class ContextLogger:
    """
    Logger with context support for structured logging.
    
    Usage:
        logger = ContextLogger(__name__)
        
        with logger.context(user_id="123", request_id="abc"):
            logger.info("Processing request")
    """
    
    def __init__(self, name: str, extra: Dict[str, Any] = None):
        self._logger = logging.getLogger(name)
        self._context: Dict[str, Any] = extra or {}
    
    def _log(self, level: int, msg: str, *args, **kwargs):
        """Log with context"""
        extra = {**self._context, **kwargs.pop("extra", {})}
        
        if extra:
            msg = f"{msg} | {json.dumps(extra)}"
        
        self._logger.log(level, msg, *args, **kwargs)
    
    def debug(self, msg: str, *args, **kwargs):
        self._log(logging.DEBUG, msg, *args, **kwargs)
    
    def info(self, msg: str, *args, **kwargs):
        self._log(logging.INFO, msg, *args, **kwargs)
    
    def warning(self, msg: str, *args, **kwargs):
        self._log(logging.WARNING, msg, *args, **kwargs)
    
    def error(self, msg: str, *args, **kwargs):
        self._log(logging.ERROR, msg, *args, **kwargs)
    
    def critical(self, msg: str, *args, **kwargs):
        self._log(logging.CRITICAL, msg, *args, **kwargs)
    
    def exception(self, msg: str, *args, **kwargs):
        self._log(logging.ERROR, msg, *args, exc_info=True, **kwargs)
    
    def context(self, **kwargs) -> "ContextLogger":
        """Create a new logger with additional context"""
        new_context = {**self._context, **kwargs}
        return ContextLogger(self._logger.name, new_context)
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass


class PerformanceLogger:
    """
    Logger for performance metrics.
    
    Usage:
        perf_logger = PerformanceLogger(__name__)
        
        with perf_logger.track("llm_call"):
            result = llm.generate(...)
        
        # Or as decorator
        @perf_logger.measure("vibe_generation")
        def generate_vibes():
            ...
    """
    
    def __init__(self, name: str):
        self._logger = logging.getLogger(name)
        self._metrics: Dict[str, list] = {}
    
    def track(self, operation: str):
        """Context manager for tracking operation time"""
        return _OperationTimer(self, operation)
    
    def record(self, operation: str, duration_ms: float, success: bool = True):
        """Record a metric"""
        if operation not in self._metrics:
            self._metrics[operation] = []
        
        self._metrics[operation].append({
            "duration_ms": duration_ms,
            "success": success,
            "timestamp": time.time()
        })
        
        # Log
        self._logger.info(
            f"{operation} completed in {duration_ms:.2f}ms "
            f"(success={success})"
        )
    
    def measure(self, operation: str):
        """Decorator for measuring function execution time"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start = time.time()
                success = True
                
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    success = False
                    raise
                finally:
                    duration_ms = (time.time() - start) * 1000
                    self.record(operation, duration_ms, success)
            
            return wrapper
        return decorator
    
    def get_stats(self, operation: str = None) -> Dict:
        """Get performance statistics"""
        if operation:
            metrics = self._metrics.get(operation, [])
        else:
            metrics = [m for ops in self._metrics.values() for m in ops]
        
        if not metrics:
            return {}
        
        durations = [m["duration_ms"] for m in metrics]
        successes = [m["success"] for m in metrics]
        
        return {
            "count": len(metrics),
            "avg_ms": sum(durations) / len(durations),
            "min_ms": min(durations),
            "max_ms": max(durations),
            "success_rate": sum(successes) / len(successes)
        }


class _OperationTimer:
    """Context manager for timing operations"""
    
    def __init__(self, logger: PerformanceLogger, operation: str):
        self.logger = logger
        self.operation = operation
        self.start_time = None
        self.success = True
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (time.time() - self.start_time) * 1000
        self.success = exc_type is None
        self.logger.record(self.operation, duration_ms, self.success)
        return False  # Don't suppress exceptions


def setup_logging(
    level: str = "INFO",
    log_file: str = None,
    json_format: bool = False
):
    """
    Setup logging configuration.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path for log output
        json_format: Use JSON format for logs
    """
    handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level.upper()))
    
    if json_format:
        formatter = logging.Formatter(
            '{"time": "%(asctime)s", "level": "%(levelname)s", '
            '"logger": "%(name)s", "message": "%(message)s"}'
        )
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    console_handler.setFormatter(formatter)
    handlers.append(console_handler)
    
    # File handler
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        handlers=handlers
    )


def get_logger(name: str) -> ContextLogger:
    """Get a context logger"""
    return ContextLogger(name)


def get_perf_logger(name: str) -> PerformanceLogger:
    """Get a performance logger"""
    return PerformanceLogger(name)
