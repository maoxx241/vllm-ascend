PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS pack_meta (
  meta_key TEXT PRIMARY KEY,
  value_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sources (
  source_id TEXT PRIMARY KEY,
  source_kind TEXT NOT NULL,
  path TEXT,
  uri TEXT,
  repo_sha TEXT,
  paired_vllm_ref TEXT,
  shard_family TEXT NOT NULL,
  excerpt_hash TEXT,
  metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS entities (
  entity_id TEXT PRIMARY KEY,
  entity_type TEXT NOT NULL,
  canonical_name TEXT NOT NULL,
  aliases_json TEXT NOT NULL DEFAULT '[]',
  tags_json TEXT NOT NULL DEFAULT '[]',
  metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS facts (
  fact_id TEXT PRIMARY KEY,
  subject_id TEXT NOT NULL,
  predicate TEXT NOT NULL,
  object_id TEXT,
  literal_text TEXT,
  confidence REAL NOT NULL DEFAULT 1.0,
  valid_from TEXT,
  valid_to TEXT,
  scope_json TEXT NOT NULL DEFAULT '{}',
  source_id TEXT NOT NULL,
  shard_family TEXT NOT NULL,
  metadata_json TEXT NOT NULL DEFAULT '{}',
  FOREIGN KEY(subject_id) REFERENCES entities(entity_id),
  FOREIGN KEY(object_id) REFERENCES entities(entity_id),
  FOREIGN KEY(source_id) REFERENCES sources(source_id)
);

CREATE TABLE IF NOT EXISTS edges (
  edge_id TEXT PRIMARY KEY,
  src_entity_id TEXT NOT NULL,
  edge_type TEXT NOT NULL,
  dst_entity_id TEXT NOT NULL,
  weight REAL NOT NULL DEFAULT 1.0,
  source_id TEXT,
  metadata_json TEXT NOT NULL DEFAULT '{}',
  FOREIGN KEY(src_entity_id) REFERENCES entities(entity_id),
  FOREIGN KEY(dst_entity_id) REFERENCES entities(entity_id),
  FOREIGN KEY(source_id) REFERENCES sources(source_id)
);

CREATE TABLE IF NOT EXISTS symbol_index (
  symbol_id TEXT PRIMARY KEY,
  qualname TEXT NOT NULL,
  kind TEXT NOT NULL,
  file_path TEXT NOT NULL,
  signature TEXT,
  owner_module TEXT,
  repo_path TEXT,
  paired_vllm_ref TEXT,
  metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS validations (
  validation_id TEXT PRIMARY KEY,
  target_id TEXT NOT NULL,
  target_kind TEXT NOT NULL,
  mode TEXT NOT NULL,
  result TEXT NOT NULL,
  env_json TEXT NOT NULL DEFAULT '{}',
  artifact_refs_json TEXT NOT NULL DEFAULT '[]',
  summary TEXT,
  source_id TEXT,
  metadata_json TEXT NOT NULL DEFAULT '{}',
  FOREIGN KEY(source_id) REFERENCES sources(source_id)
);

CREATE TABLE IF NOT EXISTS capsules (
  capsule_id TEXT PRIMARY KEY,
  request_id TEXT NOT NULL,
  intent TEXT NOT NULL,
  logical_domains_json TEXT NOT NULL DEFAULT '[]',
  selectors_json TEXT NOT NULL DEFAULT '{}',
  token_estimate INTEGER NOT NULL DEFAULT 0,
  capsule_type TEXT NOT NULL,
  capsule_text TEXT NOT NULL,
  atoms_json TEXT NOT NULL DEFAULT '[]',
  unknowns_json TEXT NOT NULL DEFAULT '[]',
  created_at TEXT NOT NULL,
  metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_entities_type_name
  ON entities(entity_type, canonical_name);

CREATE INDEX IF NOT EXISTS idx_facts_subject_predicate
  ON facts(subject_id, predicate);

CREATE INDEX IF NOT EXISTS idx_facts_object
  ON facts(object_id);

CREATE INDEX IF NOT EXISTS idx_facts_shard_family
  ON facts(shard_family);

CREATE INDEX IF NOT EXISTS idx_edges_src_type
  ON edges(src_entity_id, edge_type);

CREATE INDEX IF NOT EXISTS idx_symbol_index_qualname
  ON symbol_index(qualname);

CREATE INDEX IF NOT EXISTS idx_symbol_index_file_path
  ON symbol_index(file_path);

CREATE INDEX IF NOT EXISTS idx_validations_target
  ON validations(target_id, target_kind, result);

CREATE INDEX IF NOT EXISTS idx_capsules_request
  ON capsules(request_id, intent, created_at);
