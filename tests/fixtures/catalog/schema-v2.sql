CREATE TABLE schema_version (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    version INTEGER NOT NULL
);
CREATE TABLE media (
    id INTEGER PRIMARY KEY,
    asset_id TEXT NOT NULL UNIQUE,
    kind TEXT NOT NULL,
    path TEXT NOT NULL UNIQUE,
    size INTEGER,
    mtime_ns INTEGER,
    inode INTEGER,
    camera TEXT,
    duration TEXT,
    frame_rate TEXT,
    start_timecode TEXT,
    capture_time TEXT,
    capture_time_source TEXT,
    gps_lat REAL,
    gps_lon REAL,
    day INTEGER,
    place TEXT,
    fcp_event TEXT,
    fcp_event_uid TEXT,
    fcp_asset_uid TEXT
);
CREATE TABLE tags (
    id INTEGER PRIMARY KEY,
    media_id INTEGER NOT NULL REFERENCES media(id),
    folder TEXT NOT NULL,
    value TEXT NOT NULL,
    start TEXT,
    duration TEXT,
    source TEXT NOT NULL,
    confidence REAL,
    status TEXT NOT NULL
);
CREATE UNIQUE INDEX tags_unique ON tags(media_id, folder, ifnull(start, ''), ifnull(duration, ''));
CREATE TABLE transcripts (
    id INTEGER PRIMARY KEY,
    media_id INTEGER NOT NULL REFERENCES media(id),
    path TEXT NOT NULL,
    sha256 TEXT,
    language TEXT,
    engine TEXT,
    created TEXT
);
CREATE TABLE runs (
    id INTEGER PRIMARY KEY,
    step TEXT NOT NULL,
    started TEXT NOT NULL,
    finished TEXT,
    dry_run INTEGER NOT NULL,
    outcome TEXT
);
CREATE TABLE exports (
    id INTEGER PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    path TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    read_at TEXT NOT NULL
);
