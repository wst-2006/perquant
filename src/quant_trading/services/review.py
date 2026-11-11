from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from ..db import ReviewJournal, session_scope


@dataclass
class ReviewService:
    def add_entry(
        self,
        title: str,
        content: str,
        trade_date: date | None = None,
        ts_code: str | None = None,
        name: str | None = None,
        category: str = "daily",
        tags: list[str] | None = None,
    ) -> int:
        if not title.strip():
            raise ValueError("title is required")
        if not content.strip():
            raise ValueError("content is required")
        with session_scope() as session:
            entry = ReviewJournal(
                trade_date=trade_date or date.today(),
                ts_code=ts_code,
                name=name,
                category=category,
                title=title.strip(),
                content=content.strip(),
                tags=tags or [],
            )
            session.add(entry)
            session.flush()
            return int(entry.id)

    def list_entries(self, limit: int = 50, ts_code: str | None = None) -> list[dict]:
        with session_scope() as session:
            query = session.query(ReviewJournal)
            if ts_code:
                query = query.filter(ReviewJournal.ts_code == ts_code)
            rows = query.order_by(ReviewJournal.trade_date.desc(), ReviewJournal.created_at.desc()).limit(limit).all()
            return [
                {
                    "id": row.id,
                    "trade_date": row.trade_date.isoformat(),
                    "ts_code": row.ts_code,
                    "name": row.name,
                    "category": row.category,
                    "title": row.title,
                    "content": row.content,
                    "tags": row.tags,
                    "created_at": row.created_at.isoformat(),
                }
                for row in rows
            ]
