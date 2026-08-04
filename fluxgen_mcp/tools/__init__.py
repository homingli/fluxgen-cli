"""Tool implementations for the fluxgen MCP server.

Each module exposes one async function that takes an
`MCPSettings` + `AuditLog` + `ConcurrencyGate` and the tool's
declared arguments. The server wires those in.
"""
