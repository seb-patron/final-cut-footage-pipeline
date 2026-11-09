"""JSON schema subset check and answer validation for AI steps.

Only the standard library is used here. :func:`check_schema` accepts the
small strict subset both providers can enforce (every object lists all of
its properties in ``required`` and sets ``additionalProperties`` to
false); :func:`validate` checks a backend answer against such a schema
and returns one ``path: rule`` problem per mismatch, never the value
itself.
"""

from __future__ import annotations


class AiSchemaError(ValueError):
    """Raised when a schema leaves the supported strict subset."""


#: JSON types a schema node may name.
SUPPORTED_TYPES = (
    "object",
    "array",
    "string",
    "integer",
    "number",
    "boolean",
    "null",
)
#: Keys a schema node may use; anything else is rejected.
SUPPORTED_KEYS = frozenset(
    {
        "type",
        "properties",
        "required",
        "additionalProperties",
        "items",
        "enum",
        "minimum",
        "maximum",
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
        "description",
        "title",
    }
)


def _type_list(node_types, path: str) -> list[str]:
    """Normalize a ``type`` value to a list of supported type names."""
    if isinstance(node_types, str):
        node_types = [node_types]
    if not isinstance(node_types, list) or not node_types:
        raise AiSchemaError(f"bad schema at {path}: 'type' must name a type")
    for name in node_types:
        if name not in SUPPORTED_TYPES:
            raise AiSchemaError(
                f"bad schema at {path}: unsupported type {name!r}"
            )
    return list(node_types)


def _check_node(node, path: str) -> None:
    """Recursively enforce the supported subset at one schema node."""
    if not isinstance(node, dict):
        raise AiSchemaError(f"bad schema at {path}: must be an object")
    for key in node:
        if key not in SUPPORTED_KEYS:
            raise AiSchemaError(
                f"bad schema at {path}: unsupported keyword {key!r}"
            )
    node_type = node.get("type")
    types: list[str] | None = None
    if node_type is not None:
        types = _type_list(node_type, path)
    if "additionalProperties" in node and node["additionalProperties"] is not False:
        raise AiSchemaError(
            f"bad schema at {path}: 'additionalProperties' must be false"
        )
    required = node.get("required", None)
    if required is not None:
        if (
            not isinstance(required, list)
            or not all(isinstance(name, str) for name in required)
        ):
            raise AiSchemaError(
                f"bad schema at {path}: 'required' must list property names"
            )
    properties = node.get("properties", None)
    if properties is not None:
        if not isinstance(properties, dict):
            raise AiSchemaError(
                f"bad schema at {path}: 'properties' must be an object"
            )
        for name, sub in properties.items():
            _check_node(sub, f"{path}.{name}")
    if "items" in node:
        if not isinstance(node["items"], dict):
            raise AiSchemaError(
                f"bad schema at {path}: 'items' must be a schema"
            )
        _check_node(node["items"], f"{path}[]")
    if "enum" in node and not isinstance(node["enum"], list):
        raise AiSchemaError(f"bad schema at {path}: 'enum' must be a list")
    for key in ("minimum", "maximum"):
        if key in node and (
            isinstance(node[key], bool) or not isinstance(node[key], (int, float))
        ):
            raise AiSchemaError(
                f"bad schema at {path}: {key!r} must be a number"
            )
    for key in ("minLength", "maxLength", "minItems", "maxItems"):
        if key in node and (
            isinstance(node[key], bool)
            or not isinstance(node[key], int)
            or node[key] < 0
        ):
            raise AiSchemaError(
                f"bad schema at {path}: {key!r} must be a non-negative integer"
            )
    # Strict object form: every object lists all its properties in
    # ``required`` and forbids the rest. An optional value is typed as a
    # list holding "null" instead of being left out of ``required``.
    if (types is not None and "object" in types) or (
        types is None and properties is not None
    ):
        names = set(properties or {})
        if set(required or []) != names:
            raise AiSchemaError(
                f"bad schema at {path}: 'required' must list every property"
            )
        if node.get("additionalProperties", None) is not False:
            raise AiSchemaError(
                f"bad schema at {path}: 'additionalProperties' must be false"
            )


def check_schema(schema) -> None:
    """Raise :class:`AiSchemaError` unless ``schema`` is in the subset.

    The root must be an object; every object anywhere in the schema must
    use the strict form. Nothing is sent before this passes.
    """
    if not isinstance(schema, dict):
        raise AiSchemaError("bad schema at $: must be an object")
    _check_node(schema, "$")
    root_type = schema.get("type")
    if isinstance(root_type, str):
        root_type = [root_type]
    if not isinstance(root_type, list) or "object" not in root_type:
        raise AiSchemaError("bad schema at $: the root must be an object")


def _matches_type(value, name: str) -> bool:
    """Return whether ``value`` has the JSON Schema type ``name``."""
    if name == "object":
        return isinstance(value, dict)
    if name == "array":
        return isinstance(value, list)
    if name == "string":
        return isinstance(value, str)
    if name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if name == "boolean":
        return isinstance(value, bool)
    if name == "null":
        return value is None
    return False


def _enum_contains(value, choices: list) -> bool:
    """Membership test that never confuses booleans with numbers."""
    for choice in choices:
        if isinstance(value, bool) or isinstance(choice, bool):
            if type(value) is type(choice) and value == choice:
                return True
        elif value == choice and type(value) is type(choice):
            return True
        elif (
            isinstance(value, (int, float))
            and isinstance(choice, (int, float))
            and value == choice
        ):
            return True
    return False


def _validate_node(value, node: dict, path: str, problems: list[str]) -> None:
    """Append one ``path: rule`` problem per mismatch; never the value."""
    node_type = node.get("type")
    types: list[str] | None = None
    if node_type is not None:
        types = [node_type] if isinstance(node_type, str) else list(node_type)
    if types is not None and not any(
        _matches_type(value, name) for name in types
    ):
        if len(types) == 1:
            problems.append(f"{path}: expected {types[0]}")
        else:
            problems.append(f"{path}: expected one of {', '.join(types)}")
        return
    if "enum" in node and not _enum_contains(value, node["enum"]):
        problems.append(f"{path}: not one of the allowed values")
    if isinstance(value, dict):
        properties = node.get("properties") or {}
        required = node.get("required") or []
        for name in required:
            if name not in value:
                problems.append(f"{path}.{name}: missing required property")
        if node.get("additionalProperties", None) is False:
            for name in value:
                if name not in properties:
                    problems.append(f"{path}.{name}: unexpected property")
        for name, sub in properties.items():
            if name in value:
                _validate_node(value[name], sub, f"{path}.{name}", problems)
    elif isinstance(value, list):
        if "minItems" in node and len(value) < node["minItems"]:
            problems.append(f"{path}: too few items")
        if "maxItems" in node and len(value) > node["maxItems"]:
            problems.append(f"{path}: too many items")
        if "items" in node:
            for index, entry in enumerate(value):
                _validate_node(entry, node["items"], f"{path}[{index}]", problems)
    elif isinstance(value, str):
        if "minLength" in node and len(value) < node["minLength"]:
            problems.append(f"{path}: too short")
        if "maxLength" in node and len(value) > node["maxLength"]:
            problems.append(f"{path}: too long")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in node and value < node["minimum"]:
            problems.append(f"{path}: below minimum")
        if "maximum" in node and value > node["maximum"]:
            problems.append(f"{path}: above maximum")


def validate(value, schema: dict) -> list[str]:
    """Check ``value`` against ``schema``; return problems, empty when valid.

    Each problem names the JSON path and the rule, for example
    ``"$.clips[0].reason: missing required property"``. A boolean is
    never an integer or a number.
    """
    problems: list[str] = []
    _validate_node(value, schema, "$", problems)
    return problems
