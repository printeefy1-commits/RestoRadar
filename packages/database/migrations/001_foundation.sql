CREATE EXTENSION IF NOT EXISTS btree_gist;

CREATE TABLE restaurants (
  id uuid PRIMARY KEY,
  name text NOT NULL,
  version integer NOT NULL DEFAULT 0 CHECK (version >= 0),
  policy jsonb,
  last_heartbeat timestamptz
);
CREATE TABLE users (
  id uuid PRIMARY KEY,
  email text UNIQUE NOT NULL CHECK (email = lower(email)),
  name text NOT NULL,
  password_hash text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE memberships (
  restaurant_id uuid NOT NULL REFERENCES restaurants(id),
  user_id uuid NOT NULL REFERENCES users(id),
  role text NOT NULL CHECK (role IN ('OWNER', 'MANAGER', 'WAITER')),
  PRIMARY KEY (restaurant_id, user_id)
);
CREATE TABLE auth_sessions (
  token_hash text PRIMARY KEY,
  user_id uuid NOT NULL REFERENCES users(id),
  expires_at timestamptz NOT NULL
);
CREATE INDEX auth_sessions_expiry ON auth_sessions(expires_at);

CREATE TABLE dining_tables (id uuid PRIMARY KEY, restaurant_id uuid NOT NULL REFERENCES restaurants(id), data jsonb NOT NULL, UNIQUE (restaurant_id, id));
CREATE UNIQUE INDEX unique_table_name ON dining_tables(restaurant_id, lower(data->>'name'));
CREATE TABLE occupancy_sessions (id uuid PRIMARY KEY, restaurant_id uuid NOT NULL REFERENCES restaurants(id), data jsonb NOT NULL, UNIQUE (restaurant_id, id));
CREATE TABLE session_assignments (
  id uuid PRIMARY KEY,
  restaurant_id uuid NOT NULL REFERENCES restaurants(id),
  session_id uuid NOT NULL,
  table_id uuid NOT NULL,
  started_at timestamptz NOT NULL,
  ended_at timestamptz,
  data jsonb NOT NULL,
  CHECK (ended_at IS NULL OR ended_at >= started_at),
  FOREIGN KEY (restaurant_id, session_id) REFERENCES occupancy_sessions(restaurant_id, id),
  FOREIGN KEY (restaurant_id, table_id) REFERENCES dining_tables(restaurant_id, id),
  EXCLUDE USING gist (restaurant_id WITH =, table_id WITH =, tstzrange(started_at, ended_at, '[)') WITH &&)
);
CREATE UNIQUE INDEX one_active_session_per_table ON session_assignments(restaurant_id, table_id) WHERE ended_at IS NULL;
CREATE TABLE cleaning_cycles (id uuid PRIMARY KEY, restaurant_id uuid NOT NULL REFERENCES restaurants(id), data jsonb NOT NULL);
CREATE UNIQUE INDEX one_unresolved_cycle ON cleaning_cycles(restaurant_id, (data->>'tableId')) WHERE data->>'readyAt' IS NULL;
CREATE TABLE waiter_tasks (id uuid PRIMARY KEY, restaurant_id uuid NOT NULL REFERENCES restaurants(id), data jsonb NOT NULL);
CREATE UNIQUE INDEX unique_task_source ON waiter_tasks(restaurant_id, (data->>'kind'), (data->>'sourceId'));
CREATE TABLE assignment_attempts (id uuid PRIMARY KEY, restaurant_id uuid NOT NULL REFERENCES restaurants(id), data jsonb NOT NULL);
CREATE TABLE manager_alerts (id uuid PRIMARY KEY, restaurant_id uuid NOT NULL REFERENCES restaurants(id), data jsonb NOT NULL);
CREATE UNIQUE INDEX unique_alert_key ON manager_alerts(restaurant_id, (data->>'dedupKey'));
CREATE TABLE session_adjustments (id uuid PRIMARY KEY, restaurant_id uuid NOT NULL REFERENCES restaurants(id), data jsonb NOT NULL);
CREATE TABLE monitoring_gaps (id uuid PRIMARY KEY, restaurant_id uuid NOT NULL REFERENCES restaurants(id), data jsonb NOT NULL);
CREATE TABLE domain_events (
  id uuid PRIMARY KEY, restaurant_id uuid NOT NULL REFERENCES restaurants(id),
  sequence integer NOT NULL, data jsonb NOT NULL, UNIQUE(restaurant_id, sequence)
);
CREATE TABLE audit_log (id uuid PRIMARY KEY, restaurant_id uuid NOT NULL REFERENCES restaurants(id), data jsonb NOT NULL);
CREATE TABLE synchronization_outbox (
  event_id uuid PRIMARY KEY REFERENCES domain_events(id),
  restaurant_id uuid NOT NULL REFERENCES restaurants(id),
  status text NOT NULL DEFAULT 'QUEUED' CHECK (status IN ('QUEUED', 'DELIVERED')),
  delivered_at timestamptz
);
CREATE TABLE command_receipts (
  restaurant_id uuid NOT NULL REFERENCES restaurants(id),
  command_id uuid NOT NULL, fingerprint text NOT NULL,
  applied_version integer NOT NULL, created_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (restaurant_id, command_id)
);
CREATE FUNCTION protect_append_only() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  RAISE EXCEPTION 'append-only records cannot be changed';
END;
$$;
CREATE TRIGGER protect_audit BEFORE UPDATE OR DELETE ON audit_log FOR EACH ROW EXECUTE FUNCTION protect_append_only();
CREATE TRIGGER protect_events BEFORE UPDATE OR DELETE ON domain_events FOR EACH ROW EXECUTE FUNCTION protect_append_only();
CREATE TRIGGER protect_adjustments BEFORE UPDATE OR DELETE ON session_adjustments FOR EACH ROW EXECUTE FUNCTION protect_append_only();
CREATE TRIGGER protect_gaps BEFORE UPDATE OR DELETE ON monitoring_gaps FOR EACH ROW EXECUTE FUNCTION protect_append_only();
