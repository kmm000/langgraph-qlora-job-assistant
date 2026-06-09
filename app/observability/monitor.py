import json
import logging
import sqlite3
import uuid
from functools import wraps
from logging.handlers import RotatingFileHandler
from pathlib import Path
from time import perf_counter
from typing import Any, Callable


DB_PATH = "data/observability.db"
LOG_PATH = "logs/app.log"


Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
Path(LOG_PATH).parent.mkdir(parents=True, exist_ok=True)


logger = logging.getLogger("job_agent_observability")
logger.setLevel(logging.INFO)

if not logger.handlers:
    file_handler = RotatingFileHandler(
        LOG_PATH,
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


class MetricsStore:
    """使用 SQLite 持久化请求与 Agent 执行指标。"""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_database()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(
            self.db_path,
            timeout=30,
        )

    def _init_database(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT NOT NULL,
                    node_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    latency_ms REAL NOT NULL,
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS request_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    status TEXT NOT NULL,
                    total_latency_ms REAL NOT NULL,
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def record_agent(
        self,
        request_id: str,
        node_name: str,
        status: str,
        latency_ms: float,
        error_message: str | None = None,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO agent_runs (
                    request_id,
                    node_name,
                    status,
                    latency_ms,
                    error_message
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    request_id,
                    node_name,
                    status,
                    latency_ms,
                    error_message,
                ),
            )

    def record_request(
        self,
        request_id: str,
        operation: str,
        status: str,
        total_latency_ms: float,
        error_message: str | None = None,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO request_runs (
                    request_id,
                    operation,
                    status,
                    total_latency_ms,
                    error_message
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    request_id,
                    operation,
                    status,
                    total_latency_ms,
                    error_message,
                ),
            )

    def get_summary(self) -> dict[str, Any]:
        with self._connect() as connection:
            request_row = connection.execute(
                """
                SELECT
                    COUNT(*) AS total_requests,
                    SUM(
                        CASE WHEN status = 'success'
                        THEN 1 ELSE 0 END
                    ) AS successful_requests,
                    AVG(total_latency_ms) AS avg_latency_ms
                FROM request_runs
                """
            ).fetchone()

            agent_rows = connection.execute(
                """
                SELECT
                    node_name,
                    COUNT(*) AS total_calls,
                    SUM(
                        CASE WHEN status = 'success'
                        THEN 1 ELSE 0 END
                    ) AS successful_calls,
                    AVG(latency_ms) AS avg_latency_ms,
                    MAX(latency_ms) AS max_latency_ms
                FROM agent_runs
                GROUP BY node_name
                ORDER BY node_name
                """
            ).fetchall()

        total_requests = request_row[0] or 0
        successful_requests = request_row[1] or 0

        request_success_rate = (
            successful_requests / total_requests * 100
            if total_requests
            else 0.0
        )

        agents = []

        for row in agent_rows:
            total_calls = row[1] or 0
            successful_calls = row[2] or 0

            success_rate = (
                successful_calls / total_calls * 100
                if total_calls
                else 0.0
            )

            agents.append(
                {
                    "node_name": row[0],
                    "total_calls": total_calls,
                    "success_rate": round(success_rate, 2),
                    "avg_latency_ms": round(row[3] or 0, 2),
                    "max_latency_ms": round(row[4] or 0, 2),
                }
            )

        return {
            "requests": {
                "total": total_requests,
                "success_rate": round(
                    request_success_rate,
                    2,
                ),
                "avg_latency_ms": round(
                    request_row[2] or 0,
                    2,
                ),
            },
            "agents": agents,
        }

    def get_recent_runs(
        self,
        limit: int = 30,
    ) -> list[dict[str, Any]]:
        safe_limit = max(1, min(limit, 200))

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    request_id,
                    node_name,
                    status,
                    latency_ms,
                    error_message,
                    created_at
                FROM agent_runs
                ORDER BY id DESC
                LIMIT ?
                """,
                (safe_limit,),
            ).fetchall()

        return [
            {
                "request_id": row[0],
                "node_name": row[1],
                "status": row[2],
                "latency_ms": round(row[3], 2),
                "error_message": row[4],
                "created_at": row[5],
            }
            for row in rows
        ]


metrics_store = MetricsStore()


def new_request_id() -> str:
    return uuid.uuid4().hex


def observe_node(
    node_name: str,
) -> Callable:
    """装饰 LangGraph 节点，记录状态、耗时及异常。"""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(state: dict) -> dict:
            request_id = state.get(
                "request_id",
                new_request_id(),
            )

            start_time = perf_counter()

            try:
                result = func(state)

                latency_ms = (
                    perf_counter() - start_time
                ) * 1000

                trace = {
                    "node_name": node_name,
                    "status": "success",
                    "latency_ms": round(
                        latency_ms,
                        2,
                    ),
                    "error_message": None,
                }

                metrics_store.record_agent(
                    request_id=request_id,
                    node_name=node_name,
                    status="success",
                    latency_ms=latency_ms,
                )

                logger.info(
                    json.dumps(
                        {
                            "event": "agent_run",
                            "request_id": request_id,
                            **trace,
                        },
                        ensure_ascii=False,
                    )
                )

                previous_traces = result.get(
                    "node_traces",
                    state.get("node_traces", []),
                )

                return {
                    **result,
                    "request_id": request_id,
                    "node_traces": previous_traces
                    + [trace],
                }

            except Exception as exc:
                latency_ms = (
                    perf_counter() - start_time
                ) * 1000

                error_message = str(exc)[:1000]

                metrics_store.record_agent(
                    request_id=request_id,
                    node_name=node_name,
                    status="failed",
                    latency_ms=latency_ms,
                    error_message=error_message,
                )

                logger.exception(
                    json.dumps(
                        {
                            "event": "agent_run",
                            "request_id": request_id,
                            "node_name": node_name,
                            "status": "failed",
                            "latency_ms": round(
                                latency_ms,
                                2,
                            ),
                            "error_message": error_message,
                        },
                        ensure_ascii=False,
                    )
                )

                raise

        return wrapper

    return decorator