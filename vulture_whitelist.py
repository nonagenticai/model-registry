# Vulture whitelist — false-positive suppression for dead-code analysis.
#
# Vulture scans for unused code. Some parameters are required by method
# signatures (e.g. SQLAlchemy TypeDecorator overrides) but never referenced
# in the body. List them here so vulture ignores them.
#
# Usage:  vulture src/ vulture_whitelist.py

from sqlalchemy.engine.interfaces import Dialect

# -- SQLAlchemy TypeDecorator override parameters -----------------------------
# process_bind_param(self, value, dialect) and process_result_value(self, value,
# dialect) are part of the TypeDecorator contract.  The `dialect` parameter must
# appear in the signature even when the implementation doesn't use it.
dialect: Dialect

# -- structlog processor callback parameters -----------------------------------
# Structlog processors have the signature (logger, method_name, event_dict).
# The `_method_name` parameter is required by the contract but unused in
# add_trace_context() and add_service_context().
_method_name: str
