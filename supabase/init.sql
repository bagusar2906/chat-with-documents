-- Enable the pgvector extension if it's not already enabled
create extension if not exists vector;

-- Create the documents table with vector embeddings and optional metadata
create table if not exists documents (
  id uuid primary key default gen_random_uuid(),
  source text,
  content text,
  embedding vector(1536), -- Adjust the dimension based on your embedding model
  metadata jsonb
);

-- Create or replace the match_documents function for similarity search
create or replace function match_documents(
    query_embedding vector,
    match_count int,
    filter jsonb default '{}'
)
returns table (
    id uuid,
    source text,
    content text,
    metadata jsonb,
    similarity float
)
language sql
as $$
  select
    id,
    source,
    content,
    metadata,
    1 - (embedding <=> query_embedding) as similarity
  from documents
  where metadata @> filter
  order by embedding <=> query_embedding
  limit match_count;
$$;
