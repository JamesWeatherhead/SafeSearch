CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS cube;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE SCHEMA IF NOT EXISTS main;
GRANT USAGE ON SCHEMA main TO postgres;
GRANT CREATE ON SCHEMA main TO postgres;

DO $$
BEGIN
    RAISE NOTICE 'Database initialized successfully with extensions: vector, cube, uuid-ossp';
    RAISE NOTICE 'Database schema created and ready for use';
END
$$; 