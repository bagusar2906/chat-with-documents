create table if not exists documents (
    id uuid primary key default gen_random_uuid(),
    content text not null,
    metadata jsonb,
    embedding vector(1536) -- assuming OpenAI embeddings (e.g. `text-embedding-ada-002`)
);
create or replace function match_documents(
    query_embedding vector,
    match_count int,
    filter jsonb default '{}'
)
returns table (
    id uuid,
    content text,
    metadata jsonb,
    similarity float
)
language sql
as $$
  select
    id,
    content,
    metadata,
    1 - (embedding <=> query_embedding) as similarity
  from documents
  where metadata @> filter
  order by embedding <=> query_embedding
  limit match_count;
$$;
