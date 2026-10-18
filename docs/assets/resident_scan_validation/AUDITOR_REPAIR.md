# Auditor line-ending repair

Use auditor-v2.py.txt (or tools.audit_resident_scan_validation). The initial auditor.py.txt failed when a Git source already contained CRLF and the tested working file used LF. Version 2 accepts the original, normalized LF or CRLF byte form only. Each reconstructed form is SHA-256 checked against the correctness receipt. All raw worker records, launchers, source capsules, test sources and earlier archive payloads remain byte-identical.
