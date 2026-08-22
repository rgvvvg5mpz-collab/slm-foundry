"""Authentication, roles, and tenancy.

Password hashing is PBKDF2-HMAC-SHA256 from the standard library. That is a
deliberate, stated trade: it avoids a native dependency for a self-hosted tool,
and it is the piece to swap for argon2id (or an SSO integration) before this
faces the open internet. Token storage is already right — only the SHA-256 of a
bearer token is persisted, so a database dump does not hand over live sessions.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import os
import secrets

from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import ROLE_RANK, AuthSession, User, utcnow

_ITERATIONS = 240_000
TOKEN_TTL = dt.timedelta(hours=12)


def hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, _ITERATIONS)
    return f"pbkdf2_sha256${_ITERATIONS}${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, iters, salt_hex, want = stored.split("$")
        dk = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt_hex), int(iters))
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(dk.hex(), want)


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def login(s: Session, email: str, password: str) -> tuple[User, str] | None:
    user = s.execute(select(User).where(User.email == email.lower().strip())).scalar_one_or_none()
    if user is None or not user.active or not verify_password(password, user.password_hash):
        return None
    token = secrets.token_urlsafe(32)
    s.add(AuthSession(token_hash=_token_hash(token), user_id=user.id,
                      expires_at=utcnow() + TOKEN_TTL))
    user.last_seen_at = utcnow()
    s.flush()
    return user, token


def resolve_token(s: Session, token: str) -> User | None:
    sess = s.execute(
        select(AuthSession).where(AuthSession.token_hash == _token_hash(token))
    ).scalar_one_or_none()
    if sess is None or sess.revoked:
        return None
    expires = sess.expires_at
    if expires.tzinfo is None:                      # SQLite round-trips naive
        expires = expires.replace(tzinfo=dt.timezone.utc)
    if expires < utcnow():
        return None
    user = s.get(User, sess.user_id)
    if user is None or not user.active:
        return None
    user.last_seen_at = utcnow()
    return user


def revoke_token(s: Session, token: str) -> None:
    sess = s.execute(
        select(AuthSession).where(AuthSession.token_hash == _token_hash(token))
    ).scalar_one_or_none()
    if sess:
        sess.revoked = True


def has_role(user: User, minimum: str) -> bool:
    return ROLE_RANK.get(user.role, 0) >= ROLE_RANK[minimum]
