from functools import wraps
from flask import session, redirect, url_for, request, abort

def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("portal_user_id"):
            return redirect(url_for("login", next=request.path))
        return view(*args, **kwargs)
    return wrapped

def role_required(*roles):
    def decorator(view):
        @wraps(view)
        def wrapped(*args, **kwargs):
            if not session.get("portal_user_id"):
                return redirect(url_for("login", next=request.path))
            if session.get("portal_role") not in roles:
                abort(403)
            return view(*args, **kwargs)
        return wrapped
    return decorator